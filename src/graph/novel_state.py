"""小说写作流程的状态定义"""

from typing import TypedDict, Annotated
import operator


class NovelWritingState(TypedDict):

    # ---- 任务标识 ----
    task_id: str
    time_stamp:str

    # ---- 路由相关 ----
    source_agent: str          # 来源 agent 名称
    target_agent: str          # 目标 agent 名称

    # ---- 小说内容 ----
    novel_content: str         # 当前小说内容

    # ---- 抽取器相关 ----
    dynamic_prompt: str        # 调用抽取者时传入的动态提示词
    extract_result: str        # 抽取者返回的处理结果

    # ---- 审核相关 ----
    audit_result: bool         # 审核是否通过
    audit_feedback: str        # 审核反馈内容

    # ---- 历史记录 ----
    history: Annotated[list[dict], operator.add]  # 累积的操作历史