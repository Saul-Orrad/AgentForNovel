from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import create_audit_llm


class AuditAgent:
    """审核 Agent —— 负责审核小说内容并给出反馈"""

    def __init__(self, llm: BaseChatModel | None = None):
        self._llm = llm or create_audit_llm()
        self._system_prompt = self._load_prompt()

    @staticmethod
    def _load_prompt() -> str:
        prompt_path = Path(__file__).parent.parent.parent / "config" / "prompt" / "audit_agent_prompt.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "你是一位专业的小说审核编辑，负责审核小说内容并给出建设性反馈。"

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def invoke(self, state: dict) -> dict:
        """执行审核任务

        Args:
            state: 当前状态，包含 novel_content（小说内容）、task（任务描述）等

        Returns:
            dict: 更新后的状态，包含 audit_feedback（审核反馈）、
                  approved（是否通过审核）、iteration（递增后的迭代次数）
        """
        messages = [SystemMessage(content=self._system_prompt)]

        task = state.get("task", "")
        novel_content = state.get("novel_content", "")
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        user_parts = []
        if task:
            user_parts.append(f"## 创作任务\n{task}")
        user_parts.append(f"## 待审核小说内容\n{novel_content}")
        user_parts.append("请对以上小说内容进行审核，给出详细反馈。如果内容已达到要求，请在反馈中明确标注 [APPROVED]。")

        user_message = "\n\n".join(user_parts)
        messages.append(HumanMessage(content=user_message))

        response = self._llm.invoke(messages)

        feedback = response.content
        approved = "[APPROVED]" in feedback or iteration + 1 >= max_iterations

        return {
            "audit_feedback": feedback,
            "approved": approved,
            "iteration": iteration + 1,
            "max_iterations": max_iterations,
        }