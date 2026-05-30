"""Agent 工具集：RAG 检索 + 海况/视觉/定位模拟 + 报告上下文"""

import hashlib
import json
import random
from langchain_core.tools import tool

from utils.logging import logger

_rag = None

def _get_rag():
    global _rag
    if _rag is None:
        from rag.engine import RagService
        _rag = RagService()
    return _rag


def _seeded_rng(key: str) -> random.Random:
    seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return random.Random(seed)


# 最近一次海况缓存（安全联锁用）
_last_sea = {"wave": 0, "current": 0, "safe": True}


@tool(description="从海洋牧场向量知识库中检索 SOP 参考资料")
def rag_summarize(query: str) -> str:
    return _get_rag().rag_summarize(query)


@tool(description="获取指定海区实时海况（浪高、流速、风速、水温），派单前必须调用")
def get_sea_state(location: str) -> str:
    global _last_sea
    rng = _seeded_rng(location.strip())
    wave = round(rng.uniform(0.2, 3.5), 1)
    current = round(rng.uniform(0.1, 1.8), 1)
    wind = rng.choice(["偏南风2级", "东北风4级", "北风6级(阵风)", "无风"])
    temp = rng.randint(18, 26)

    safe = wave <= 2.5 and current <= 1.0
    _last_sea = {"wave": wave, "current": current, "safe": safe}

    unsafe = []
    if wave > 2.5:
        unsafe.append(f"浪高 {wave}m 超出安全阈值 2.5m")
    if current > 1.0:
        unsafe.append(f"底层流速 {current}m/s 超出安全阈值 1.0m/s")

    status = "【安全，适合下水作业】" if safe else "【禁止下水】" + "; ".join(unsafe)
    return (f"{location} 实时海况：\n"
            f"  · 浪高: {wave}m (阈值: 2.5m)\n"
            f"  · 底层流速: {current}m/s (阈值: 1.0m/s)\n"
            f"  · 风速: {wind}\n"
            f"  · 水温: {temp}℃\n"
            f"  · 综合评估: {status}")


@tool(description="获取视觉模型检测到的最新异常数据（JSON 格式）")
def invoke_vision_model(image_id: str) -> str:
    rng = _seeded_rng(image_id.strip())
    scenarios = [
        {"status": "正常", "objects": []},
        {"status": "警告", "objects": [{
            "class": "海星", "count": rng.randint(2, 8),
            "bbox": [120, 340, 200, 420], "confidence": round(rng.uniform(0.85, 0.95), 2)
        }]},
        {"status": "危险", "objects": [
            {"class": "海胆", "count": rng.randint(3, 15),
             "bbox": [80, 200, 300, 450], "confidence": round(rng.uniform(0.82, 0.92), 2)},
            {"class": "网衣破损", "severity": f"孔洞直径约{rng.randint(10, 20)}cm",
             "bbox": [400, 250, 480, 350], "confidence": round(rng.uniform(0.72, 0.85), 2)}
        ]},
    ]
    detections = rng.choice(scenarios)
    logger.info(f"[vision] {image_id}: {detections['status']}")
    return json.dumps(detections, ensure_ascii=False, indent=2)


@tool(description="自动计算水下作业人员与目标维修点的 3D 距离和当前深度")
def get_worker_nav_info() -> str:
    rng = random.Random()
    distance = round(rng.uniform(10, 40), 1)
    depth = round(rng.uniform(8, 25), 1)
    target = rng.choice(["B区侧网破损坐标", "C区底网海星聚集点", "A区锚固巡检点"])
    logger.info(f"[nav] 距 {target} {distance}m, 深度 {depth}m")
    return f"【定位信息】距目标维修点({target}) {distance} 米，当前深度 {depth} 米。"


@tool(description="汇总当前环境/感知/硬件状态，为巡检报告生成注入上下文。须先调用 get_sea_state")
def fill_context_for_report() -> str:
    if _last_sea["wave"] == 0:
        return "**【安全联锁】尚未获取海况数据，请先调用 get_sea_state。**"
    if not _last_sea["safe"]:
        return (f"**【安全联锁-工单拦截】** 当前不满足下水条件："
                f"浪高 {_last_sea['wave']}m, 流速 {_last_sea['current']}m/s。"
                f"**操作已拦截，建议改派 ROV 或延后。**")
    logger.info("[report] 海况安全确认通过")
    return "巡检上下文已就绪，请开始生成报告。"


TOOLS_ALL = [rag_summarize, get_sea_state, invoke_vision_model, get_worker_nav_info, fill_context_for_report]
