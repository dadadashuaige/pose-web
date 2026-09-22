"""可插拔引擎注册表。

三个环节各自独立，靠 key 查找实现类，pipeline 只认接口不认具体实现：

    姿态估计   POSE_ENGINES      mec | yolo11n
    行为识别   BEHAVIOR_ENGINES  none | rule | llm | stgcn
    建议生成   ADVICE_ENGINES    template | llm | rag

新增一个模型只要在对应目录实现接口并注册一行，其他地方都不用改。
"""

from __future__ import annotations

from .advice import ADVICE_ENGINES, get_advice_engine
from .behavior import BEHAVIOR_ENGINES, get_behavior_engine
from .pose import POSE_ENGINES, get_pose_engine

__all__ = [
    "POSE_ENGINES",
    "BEHAVIOR_ENGINES",
    "ADVICE_ENGINES",
    "get_pose_engine",
    "get_behavior_engine",
    "get_advice_engine",
]
