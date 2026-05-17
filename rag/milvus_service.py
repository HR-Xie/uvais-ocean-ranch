import os
import hashlib
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from milvus_model.hybrid import BGEM3EmbeddingFunction

# 导入自定义工具模块
from utils.path_tool import get_abs_path
from utils.config_handler import milvus_conf
from utils.logger_handler import logger  # 引入全局日志组件

# 导入文档处理组件
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class MilvusService:
    """
    UVAIS 海洋牧场智能运维系统 - 向量数据库服务组件
    功能：支持多格式文档解析、MD5文件去重、BGE-M3混合向量生成及Milvus增量入库。
    """

    def __init__(self):
        """
        从配置文件加载参数并初始化核心引擎
        """
        # 1. 基础配置加载
        self.uri = milvus_conf["uri"]
        self.collection_name = milvus_conf["collection_name"]
        self.vector_dim = milvus_conf["vector_dim"]
        self.model_path = get_abs_path(milvus_conf["model_path"])

        # 2. 路径配置加载
        self.data_path = get_abs_path(milvus_conf["data_path"])
        self.md5_store_path = get_abs_path(milvus_conf["md5_hex_store"])
        self.md5_store_name = os.path.basename(self.md5_store_path)

        # 3. 初始化 Milvus 客户端
        logger.info(f"正在连接 Milvus 数据库 [{self.uri}]...")
        self.client = MilvusClient(uri=self.uri)

        # 4. 初始化 BGE-M3 模型
        logger.info(f"正在加载纯本地 BGE-M3 模型 (路径: {self.model_path})...")
        self.bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=self.model_path,
            device='cpu',
            use_fp16=False
        )

        # 5. 初始化文本分片器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=milvus_conf["chunk"]["chunk_size"],
            chunk_overlap=milvus_conf["chunk"]["chunk_overlap"],
            separators=milvus_conf["chunk"]["separators"]
        )

        # 自动初始化表结构
        self._init_collection()

    def _init_collection(self):
        """
        初始化知识库表结构。如果表已存在则跳过。
        """
        if self.client.has_collection(self.collection_name):
            logger.info(f"知识库表 [{self.collection_name}] 已存在，准备就绪。")
            return

        logger.info(f"正在首次创建知识库表 [{self.collection_name}]...")
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)

        # 定义字段：ID(主键)、文本内容、来源、稠密向量、稀疏向量
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 配置双索引
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_type="FLAT", metric_type="IP")
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"表结构与双路索引创建成功: {self.collection_name}")

    # ==========================================
    # 🗂️ 文件查重机制 (MD5 策略)
    # ==========================================
    def _get_file_md5(self, file_path: str) -> str:
        """计算文件的 MD5 值"""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _check_md5_exists(self, md5_hex: str) -> bool:
        """检查 MD5 记录，确保不重复录入"""
        os.makedirs(os.path.dirname(self.md5_store_path), exist_ok=True)

        if not os.path.exists(self.md5_store_path):
            with open(self.md5_store_path, "w", encoding="utf-8") as f:
                pass
            return False

        with open(self.md5_store_path, "r", encoding="utf-8") as f:
            processed_md5s = [line.strip() for line in f.readlines()]
            return md5_hex in processed_md5s

    def _save_md5(self, md5_hex: str):
        """记录已处理文件的 MD5"""
        with open(self.md5_store_path, "a", encoding="utf-8") as f:
            f.write(md5_hex + "\n")

    def _format_sparse_vector(self, sparse_item) -> dict:
        """格式化稀疏矩阵以适配 Milvus 存储"""
        sparse_dict = {}
        if isinstance(sparse_item, dict):
            sparse_dict = {int(k): float(v) for k, v in sparse_item.items()}
        else:
            dok = sparse_item.todok()
            for coord, val in dok.items():
                key = coord[-1] if isinstance(coord, tuple) else coord
                sparse_dict[int(key)] = float(val)
        return sparse_dict

    # ==========================================
    # 📥 知识构建阶段
    # ==========================================
    def build_knowledge_base(self):
        """
        扫描配置目录，增量录入新文档
        """
        if not os.path.exists(self.data_path):
            logger.error(f"[知识构建] 数据目录不存在: {self.data_path}")
            return

        logger.info(f"[知识构建] 开始扫描数据目录: {self.data_path}")
        allowed_types = tuple(milvus_conf["allow_knowledge_file_type"])

        for file_name in os.listdir(self.data_path):
            if not file_name.lower().endswith(allowed_types):
                continue

            # 排除 MD5 记录文件本身
            if file_name == self.md5_store_name:
                logger.debug(f"[加载知识库] 跳过 MD5 记录文件: {file_name}")
                continue

            file_path = os.path.join(self.data_path, file_name)
            md5_hex = self._get_file_md5(file_path)

            if self._check_md5_exists(md5_hex):
                logger.info(f"[加载知识库] {file_name} 内容已经存在知识库内，跳过")
                continue

            logger.info(f"[加载知识库] 开始解析新文件: {file_name}")
            try:
                # 文档加载
                if file_name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding="utf-8")

                documents = loader.load()
                if not documents:
                    logger.warning(f"[加载知识库] {file_name} 内没有有效文本内容，跳过")
                    continue

                # 文本切片
                split_docs = self.text_splitter.split_documents(documents)
                if not split_docs:
                    logger.warning(f"[加载知识库] {file_name} 分片后没有有效文本内容，跳过")
                    continue

                texts = [doc.page_content for doc in split_docs]

                # 向量化 (BGE-M3 双路)
                logger.debug(f"[加载知识库] 正在对 {file_name} 提取稠密与稀疏向量...")
                embeddings = self.bge_m3_ef.encode_documents(texts)

                # 入库数据组装
                insert_data = []
                for i in range(len(texts)):
                    insert_data.append({
                        "text": texts[i],
                        "source": file_name,
                        "dense_vector": embeddings['dense'][i],
                        "sparse_vector": self._format_sparse_vector(embeddings['sparse'][i])
                    })

                self.client.insert(collection_name=self.collection_name, data=insert_data)
                self._save_md5(md5_hex)
                logger.info(f"[加载知识库] {file_name} (共 {len(texts)} 个分片) 加载成功")

            except Exception as e:
                # 开启 exc_info，将完整错误栈写入日志文件
                logger.error(f"[加载知识库] {file_name} 加载失败：{str(e)}", exc_info=True)
                continue

    # ==========================================
    # 🔄 全量重建
    # ==========================================
    def rebuild_full(self):
        """
        清空 Collection 和 MD5 记录，全量重建知识库。
        用于修复 MD5 与 Collection 不同步、或混入脏数据后的重置。
        """
        logger.info("[知识重建] 正在删除旧 Collection...")
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        logger.info(f"[知识重建] 正在清空 MD5 记录文件 ({self.md5_store_path})...")
        if os.path.exists(self.md5_store_path):
            os.remove(self.md5_store_path)

        logger.info("[知识重建] 重新初始化 Collection...")
        self._init_collection()

        logger.info("[知识重建] 开始全量重建知识库...")
        self.build_knowledge_base()
        logger.info("[知识重建] 全量重建完成。")

    # ==========================================
    # 🔍 检索阶段
    # ==========================================
    def hybrid_search(self, query: str, top_k: int = None) -> list:
        """
        执行双路混合检索与 RRF 融合
        """
        if top_k is None:
            top_k = milvus_conf["search"]["final_top_k"]

        logger.info(f"[混合检索] 接收到检索请求 -> Query: '{query}'")

        # 查询向量化
        q_embeddings = self.bge_m3_ef.encode_queries([query])

        # 组装混合搜索请求
        req_dense = AnnSearchRequest(
            data=[q_embeddings['dense'][0]],
            anns_field="dense_vector",
            param={"metric_type": "IP"},
            limit=milvus_conf["search"]["vector_top_k"]
        )

        q_sparse_dict = self._format_sparse_vector(q_embeddings['sparse'][0])
        req_sparse = AnnSearchRequest(
            data=[q_sparse_dict],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=milvus_conf["search"]["bm25_top_k"]
        )

        # RRF 融合重排
        logger.debug(f"[混合检索] 执行 RRF 融合打分 (rrf_k={milvus_conf['rrf_k']})...")
        res = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[req_dense, req_sparse],
            ranker=RRFRanker(k=milvus_conf["rrf_k"]),
            limit=top_k,
            output_fields=["text", "source"]
        )

        results = [{"text": h['entity']['text'], "source": h['entity']['source'], "score": h['distance']} for h in
                   res[0]]
        logger.info(f"[混合检索] 检索完成，成功召回 {len(results)} 条结果。")
        return results


if __name__ == '__main__':
    # 快速功能测试
    service = MilvusService()
    service.build_knowledge_base()

    # 检索测试
    test_res = service.hybrid_search("海星危害处理")
    for r in test_res:
        print(r['text'])
        logger.debug(f"召回结果: {r['source']} | 分数: {r['score']}")