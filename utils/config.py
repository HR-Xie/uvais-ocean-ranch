"""配置加载 + 路径工具"""

import os
import yaml
from dotenv import load_dotenv; load_dotenv()


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_abs_path(relative_path: str) -> str:
    return os.path.join(get_project_root(), relative_path)


def load_yaml(filename: str) -> dict:
    path = get_abs_path(filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


milvus_conf = load_yaml("config/milvus.yml")
prompts_conf = load_yaml("config/prompts.yml")
agent_conf = load_yaml("config/agent.yml")

if os.getenv("MILVUS_URI"):
    milvus_conf["uri"] = os.getenv("MILVUS_URI")
if os.getenv("MILVUS_COLLECTION_NAME"):
    milvus_conf["collection_name"] = os.getenv("MILVUS_COLLECTION_NAME")
