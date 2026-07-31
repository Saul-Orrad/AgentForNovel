"""小说写作流程的状态定义"""

from typing import TypedDict, Annotated
import operator


class NovelWritingState(TypedDict):
    """小说写作流程的状态定义"""
    task: str
    novel_content: str
    audit_feedback: str
    iteration: int
    max_iterations: int
    approved: bool
    history: Annotated[list[dict], operator.add]