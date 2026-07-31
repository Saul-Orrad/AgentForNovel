from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.apply_agent import ApplyAgent
from src.agents.audit_agent import AuditAgent
from .novel_state import NovelWritingState


class NovelWritingGraph:
    """基于 LangGraph 的小说写作图 —— ApplyAgent 与 AuditAgent 循环流转"""

    def __init__(self, apply_agent: ApplyAgent | None = None, audit_agent: AuditAgent | None = None):
        self.apply_agent = apply_agent or ApplyAgent()
        self.audit_agent = audit_agent or AuditAgent()
        self._graph = self._build_graph()

    def _apply_node(self, state: NovelWritingState) -> dict:
        """ApplyAgent 节点：撰写/修改小说内容"""
        result = self.apply_agent.invoke(state)
        result["history"] = [{"role": "apply", "iteration": state.get("iteration", 0)}]
        return result

    def _audit_node(self, state: NovelWritingState) -> dict:
        """AuditAgent 节点：审核小说内容"""
        result = self.audit_agent.invoke(state)
        result["history"] = [{"role": "audit", "iteration": state.get("iteration", 0)}]
        return result

    def _should_continue(self, state: NovelWritingState) -> Literal["apply", "__end__"]:
        """路由判断：审核通过或达到最大迭代次数则结束，否则回到 ApplyAgent 继续修改"""
        if state.get("approved", False):
            return END
        if state.get("iteration", 0) >= state.get("max_iterations", 3):
            return END
        return "apply"

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 图

        流程：
        START -> apply (撰写) -> audit (审核) -> 判断是否通过
          - 通过 / 超限 -> END
          - 未通过 -> apply (修改) -> audit (审核) -> ...循环
        """
        builder = StateGraph(NovelWritingState)

        builder.add_node("apply", self._apply_node)
        builder.add_node("audit", self._audit_node)

        builder.set_entry_point("apply")
        builder.add_edge("apply", "audit")
        builder.add_conditional_edges(
            "audit",
            self._should_continue,
            {
                "apply": "apply",
                END: END,
            },
        )

        memory = MemorySaver()
        return builder.compile(checkpointer=memory)

    @property
    def graph(self):
        return self._graph

    def run(self, task: str, max_iterations: int = 3, config: dict | None = None) -> NovelWritingState:
        """运行小说写作流程

        Args:
            task: 创作任务描述
            max_iterations: 最大循环次数
            config: LangGraph 配置（含 thread_id 等）

        Returns:
            NovelWritingState: 最终状态
        """
        initial_state: NovelWritingState = {
            "task": task,
            "novel_content": "",
            "audit_feedback": "",
            "iteration": 0,
            "max_iterations": max_iterations,
            "approved": False,
            "history": [],
        }

        if config is None:
            config = {"configurable": {"thread_id": "default"}}

        return self._graph.invoke(initial_state, config)


def create_novel_writing_graph(
    apply_agent: ApplyAgent | None = None,
    audit_agent: AuditAgent | None = None,
) -> NovelWritingGraph:
    """工厂函数：创建小说写作图实例"""
    return NovelWritingGraph(apply_agent=apply_agent, audit_agent=audit_agent)