import os

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 导入你刚刚完美重构的向量库服务和日志器
from rag.milvus_service import MilvusService
from rag.reranker_service import RerankerService
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts
from utils.config_handler import rag_conf
from model.factory import get_chat_model

class RagSummarizeService:
    """
    UVAIS 海洋牧场 RAG 总结服务类
    工作流：接收查询 -> 调用 Milvus 混合检索 -> 拼接 Context -> 提交大模型 -> 返回专业总结
    """

    def __init__(self):
        logger.info("正在初始化 RAG 总结服务...")
        self.vector_store = MilvusService()
        self.reranker = RerankerService()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = get_chat_model()
        self.chain = self._init_chain()
        logger.info("RAG 总结服务初始化完成！")

    def _init_chain(self):
        """组装 LangChain LCEL 工作流：提示词模板 -> 大模型 -> 字符串解析器"""
        # 这个写法的神奇之处在于，数据会像流水线一样自动流转
        return self.prompt_template | self.model | StrOutputParser()

    def rag_summarize(self, query: str) -> str:
        """
        对外暴露的核心调用方法
        """
        logger.info(f"[RAG 总结] 开始处理查询: '{query}'")

        # 1. Milvus 混合检索 + RRF 融合召回
        candidates = self.vector_store.hybrid_search(query)

        if not candidates:
            logger.warning("[RAG 总结] 向量库未召回任何相关资料。")
            return "很抱歉，当前的知识库中没有查找到相关记录。"

        # 2. BGE-Reranker Cross-Encoder 精排
        top_docs = self.reranker.rerank(query, candidates)

        # 3. 格式化拼接 Context
        context_str = ""
        for i, doc in enumerate(top_docs):
            context_str += f"【参考资料 {i + 1}】\n"
            context_str += f"来源：{doc['source']}\n"
            context_str += f"内容：{doc['text']}\n"
            context_str += "-" * 30 + "\n"

        logger.debug(f"[RAG 总结] 拼接后的参考资料为:\n{context_str}")

        # 4. 提交大模型生成最终结果
        logger.info("[RAG 总结] 正在等待大模型生成总结回复...")
        try:
            final_answer = self.chain.invoke({
                "input": query,
                "context": context_str
            })
            logger.info("[RAG 总结] 回复生成成功。")
            return final_answer

        except Exception as e:
            logger.error(f"[RAG 总结] 大模型调用失败: {str(e)}", exc_info=True)
            return "系统异常，暂时无法生成总结，请稍后再试。"


if __name__ == '__main__':
    # 填入你的真实 API KEY 运行测试

    rag_service = RagSummarizeService()

    test_query = "发现海星聚集怎么处理？"



    answer = rag_service.rag_summarize(test_query)
    print(answer)
    #
    # print("\n" + "=" * 40)
    # print("🤖 智能管家最终回复:")
    # print("=" * 40)
    # print(answer)