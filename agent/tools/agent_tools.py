import json
import random
from langchain_core.tools import tool

from utils.logger_handler import logger

# Lazy RAG instance to avoid heavy imports (Milvus, BGE, ChatOpenAI) at module level
_rag = None

def _get_rag():
    global _rag
    if _rag is None:
        from rag.summarize_service import RagSummarizeService
        _rag = RagSummarizeService()
    return _rag

# 模拟海域
zones = ["1号养殖区", "深水锚泊区", "繁育网箱区"]

# 缓存最近一次海况查询结果，供安全联锁使用
_last_sea_state = {"wave": None, "current": None, "safe": True}


# ==========================================
# 1. RAG 知识库检索 (全角色)
# ==========================================
@tool(description="从海洋牧场向量知识库中检索 SOP 参考资料，处理海星海胆灾害防治、设备排障、水质标准等专业问题")
def rag_summarize(query: str) -> str:
    return _get_rag().rag_summarize(query)


# ==========================================
# 2. 海况安全评估 (全角色)
# ==========================================
@tool(description="获取指定海区/网箱的实时海况（浪高、底层流速、风速、水温），用于安全评估和下水判定。Manager 派单前必须调用此工具。")
def get_sea_state(location: str) -> str:
    wave = round(random.uniform(0.2, 3.5), 1)
    current = round(random.uniform(0.1, 1.8), 1)
    wind = random.choice(["偏南风2级", "东北风4级", "北风6级(阵风)", "无风"])
    temp = random.randint(18, 26)

    unsafe = []
    if wave > 2.5:
        unsafe.append(f"浪高 {wave}m 超出安全阈值 2.5m")
    if current > 1.0:
        unsafe.append(f"底层流速 {current}m/s 超出安全阈值 1.0m/s")

    is_safe = len(unsafe) == 0

    # 缓存供 fill_context_for_report 安全联锁使用
    _last_sea_state["wave"] = wave
    _last_sea_state["current"] = current
    _last_sea_state["safe"] = is_safe

    if is_safe:
        status = "【安全，适合下水作业】"
    else:
        status = "【禁止下水】" + "; ".join(unsafe)

    return (f"{location} 实时海况：\n"
            f"  · 浪高: {wave}m (安全阈值: 2.5m)\n"
            f"  · 底层流速: {current}m/s (安全阈值: 1.0m/s)\n"
            f"  · 风速: {wind}\n"
            f"  · 水温: {temp}℃\n"
            f"  · 综合评估: {status}")


# ==========================================
# 3. 视觉感知 (全角色)
# ==========================================
@tool(description="获取 SEAD-YOLO 视觉模型检测到的最新异常数据，返回 JSON 格式的目标类别、数量、位置及置信度")
def invoke_vision_model(image_id: str) -> str:
    detections = random.choice([
        {"status": "正常", "objects": []},
        {"status": "警告", "objects": [
            {"class": "海星", "count": random.randint(2, 8),
             "bbox": [120, 340, 200, 420], "confidence": 0.92}
        ]},
        {"status": "危险", "objects": [
            {"class": "海胆", "count": random.randint(3, 15),
             "bbox": [80, 200, 300, 450], "confidence": 0.87},
            {"class": "网衣破损", "severity": "孔洞直径约15cm",
             "bbox": [400, 250, 480, 350], "confidence": 0.78}
        ]},
    ])
    logger.info(f"[视觉感知] 设备 {image_id} 检测结果: {detections['status']}")
    return json.dumps(detections, ensure_ascii=False, indent=2)


# ==========================================
# 4. Worker 自动定位测距 (worker_ 专属)
# ==========================================
@tool(description="自动计算并返回水下作业人员与目标维修点的 3D 距离和当前深度。Worker 提问时自动触发。")
def get_worker_nav_info() -> str:
    distance = round(random.uniform(3.0, 80.0), 1)
    depth = round(random.uniform(5.0, 35.0), 1)
    target = random.choice(["B区侧网破损坐标", "C区底网海星聚集点", "A区锚固巡检点"])
    logger.info(f"[定位] 距目标 {distance}m, 深度 {depth}m")
    return f"【定位信息】距目标维修点({target}) {distance} 米，当前深度 {depth} 米。"


# ==========================================
# 5. 巡检报告上下文注入 (manager_ 专属)
# ==========================================
@tool(description="汇总当前环境数据、感知数据、硬件状态，为巡检报告生成注入全局上下文。Manager 专属，必须先调用 get_sea_state。")
def fill_context_for_report() -> str:
    # 硬拦截：未查海况或海况不安全时拒绝生成工单
    if _last_sea_state["wave"] is None:
        return "**【安全联锁】尚未获取海况数据，请先调用 get_sea_state 评估作业环境。**"

    if not _last_sea_state["safe"]:
        return (
            f"**【安全联锁-工单拦截】**\n"
            f"当前海况不满足下水作业条件：浪高 {_last_sea_state['wave']}m (阈值 2.5m)，"
            f"底层流速 {_last_sea_state['current']}m/s (阈值 1.0m/s)。\n"
            f"**操作已拦截，工单不予生成。建议改派 ROV 或延后至海况好转。**"
        )

    logger.info("📄 [中间件触发] 海况安全确认通过，正在生成工单上下文...")
    return "巡检上下文已就绪，请开始生成报告。"


# ==========================================
# 📦 工具箱导出
# ==========================================
TOOLS_ALL = [rag_summarize, get_sea_state, invoke_vision_model, get_worker_nav_info, fill_context_for_report]
