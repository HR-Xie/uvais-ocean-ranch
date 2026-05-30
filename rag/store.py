"""知识库存储：文档加载 + 分片 + Milvus 索引 + MD5 去重"""

import os
import hashlib

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pymilvus import connections, utility

from utils.config import milvus_conf, get_abs_path
from utils.logging import logger


class KnowledgeBase:

    def __init__(self):
        cfg = milvus_conf
        self._data_path = get_abs_path(cfg["data_path"])
        self._md5_path = get_abs_path(cfg["md5_hex_store"])
        self._collection = cfg["collection_name"]
        self._uri = cfg["uri"]
        self._allow_exts = [f".{t}" for t in cfg["allow_knowledge_file_type"]]

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=get_abs_path(cfg["model_path"]), device="cpu"
        )
        self._splitter = SentenceSplitter(
            chunk_size=cfg["chunk"]["chunk_size"],
            chunk_overlap=cfg["chunk"]["chunk_overlap"],
        )
        self._nodes: list = []
        self._index: VectorStoreIndex = None

    # ── 索引 ─────────────────────────────────────────────────────

    def _get_vector_store(self, overwrite: bool = False) -> MilvusVectorStore:
        vs = MilvusVectorStore(
            uri=self._uri, collection_name=self._collection,
            dim=milvus_conf["vector_dim"], overwrite=overwrite,
        )
        # patch aquery → 同步，避免嵌套 asyncio 冲突
        async def _patched_aquery(query, **kw):
            return vs.query(query, **kw)
        vs.__dict__["aquery"] = _patched_aquery
        return vs

    def _collection_exists(self) -> bool:
        try:
            connections.connect(alias="default", uri=self._uri)
            return utility.has_collection(self._collection, using="default")
        except Exception:
            return False

    def _load_docs(self, file_paths: list[str] | None = None) -> list:
        """加载文档并分片。file_paths=None 时加载整个 data 目录。"""
        if file_paths:
            reader = SimpleDirectoryReader(input_files=file_paths)
        else:
            reader = SimpleDirectoryReader(
                input_dir=self._data_path, required_exts=self._allow_exts, recursive=False
            )
        docs = reader.load_data()
        if not docs:
            return []
        nodes = self._splitter.get_nodes_from_documents(docs)
        logger.info(f"[KB] {len(docs)} docs → {len(nodes)} nodes")
        return nodes

    def _init_index(self) -> VectorStoreIndex:
        if self._index is not None:
            return self._index
        vs = self._get_vector_store(overwrite=False)
        self._index = VectorStoreIndex.from_vector_store(vs)
        if not self._nodes:
            self._nodes = self._load_docs()
        return self._index

    def get_index(self) -> VectorStoreIndex:
        if self._index is None:
            return self._init_index()
        return self._index

    def get_nodes(self) -> list:
        if not self._nodes:
            self._init_index()
        return self._nodes

    # ── 入库 ─────────────────────────────────────────────────────

    def build(self) -> int:
        """增量入库"""
        new_files = self._list_new_files()
        if not new_files:
            logger.info("[KB] 无新文件")
            self._init_index()
            return 0

        logger.info(f"[KB] {len(new_files)} new files")
        new_nodes = self._load_docs(new_files)
        if not new_nodes:
            return 0

        if self._collection_exists():
            self._init_index()
            self._index.insert_nodes(new_nodes)
            if self._nodes:
                self._nodes.extend(new_nodes)
        else:
            vs = self._get_vector_store(overwrite=True)
            sc = StorageContext.from_defaults(vector_store=vs)
            self._index = VectorStoreIndex(nodes=new_nodes, storage_context=sc)
            vs.client.flush(self._collection)
            self._nodes = new_nodes

        for f in new_files:
            self._mark_processed(f)
        logger.info(f"[KB] done: +{len(new_nodes)} nodes, total {len(self._nodes)}")
        return len(new_nodes)

    def rebuild(self) -> int:
        """全量重建"""
        logger.info("[KB] 全量重建...")
        nodes = self._load_docs()
        if not nodes:
            return 0
        vs = self._get_vector_store(overwrite=True)
        sc = StorageContext.from_defaults(vector_store=vs)
        self._index = VectorStoreIndex(nodes=nodes, storage_context=sc)
        vs.client.flush(self._collection)
        self._nodes = nodes
        self._clear_md5()
        for f in os.listdir(self._data_path):
            if f.lower().endswith(tuple(self._allow_exts)):
                self._mark_processed(os.path.join(self._data_path, f))
        logger.info(f"[KB] rebuild done: {len(nodes)} nodes")
        return len(nodes)

    # ── MD5 去重 ─────────────────────────────────────────────────

    @staticmethod
    def _md5(file_path: str) -> str:
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    def _list_new_files(self) -> list[str]:
        if not os.path.exists(self._data_path):
            return []
        md5_fname = os.path.basename(self._md5_path)
        existing = set()
        if os.path.exists(self._md5_path):
            with open(self._md5_path, "r", encoding="utf-8") as f:
                existing = {l.strip() for l in f}
        new = []
        for fname in os.listdir(self._data_path):
            if fname == md5_fname:
                continue
            fpath = os.path.join(self._data_path, fname)
            if fname.lower().endswith(tuple(self._allow_exts)) and self._md5(fpath) not in existing:
                new.append(fpath)
        return new

    def _mark_processed(self, file_path: str):
        md5 = self._md5(file_path)
        with open(self._md5_path, "a", encoding="utf-8") as f:
            f.write(md5 + "\n")

    def _clear_md5(self):
        if os.path.exists(self._md5_path):
            os.remove(self._md5_path)
