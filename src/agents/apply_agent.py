from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import create_apply_llm


class ApplyAgent:
    """写作 Agent —— 负责撰写和修改小说内容"""

    def __init__(self, llm: BaseChatModel | None = None):
        self._llm = llm or create_apply_llm()
        self._system_prompt = self._load_prompt()

    @staticmethod
    def _load_prompt() -> str:
        prompt_path = Path(__file__).parent.parent.parent / "config" / "prompt" / "apply_agent_prompt.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "你是一位专业的小说写作助手，负责撰写小说内容。"

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def invoke(self, state: dict) -> dict:
        """执行写作任务

        Args:
            state: 当前状态，包含 novel_content（已有小说内容）、
                   audit_feedback（审核反馈）、task（用户任务描述）等

        Returns:
            dict: 更新后的状态，包含 novel_content（撰写/修改后的小说内容）
        """
        messages = [SystemMessage(content=self._system_prompt)]

        task = state.get("task", "")
        novel_content = state.get("novel_content", "")
        audit_feedback = state.get("audit_feedback", "")
        max_iterations = state.get("max_iterations", 3)
        iteration = state.get("iteration", 0)

        # 构建用户消息
        user_parts = []
        if task:
            user_parts.append(f"## 创作任务\n{task}")
        if novel_content:
            user_parts.append(f"## 当前小说内容\n{novel_content}")
        if audit_feedback:
            user_parts.append(f"## 审核反馈\n{audit_feedback}\n\n请根据审核反馈修改小说内容。")
        else:
            user_parts.append("请开始撰写小说内容。")

        user_message = "\n\n".join(user_parts)
        messages.append(HumanMessage(content=user_message))

        response = self._llm.invoke(messages)

        return {
            "novel_content": response.content,
            "iteration": iteration,
            "max_iterations": max_iterations,
        }