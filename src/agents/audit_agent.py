from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import create_audit_llm


class AuditAgent:
    """审核节点 —— 对小说内容进行审核，决定是否通过或需要回退到 detail"""

    def __init__(self, llm: BaseChatModel | None = None):
        self._llm = llm or create_audit_llm()
        self._system_prompt = self._load_prompt()

    @staticmethod
    def _load_prompt() -> str:
        prompt_path = Path(__file__).parent.parent.parent / "config" / "prompt" / "audit_prompt.yml"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "对小说内容进行审核，给出详细反馈。如果内容已达到要求，请明确标注 [APPROVED]。"

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def invoke(self, state: dict) -> dict:
        messages = [SystemMessage(content=self._system_prompt)]

        novel_content = state.get("novel_content", "")

        user_parts = [f"## 待审核小说内容\n{novel_content}"]
        user_parts.append("请对以上小说内容进行审核，给出详细反馈。如果内容已达到要求，请在反馈中明确标注 [APPROVED]。")

        user_message = "\n\n".join(user_parts)
        messages.append(HumanMessage(content=user_message))

        response = self._llm.invoke(messages)
        feedback = response.content
        approved = "[APPROVED]" in feedback

        return {
            "audit_result": approved,
            "audit_feedback": feedback,
            "source_agent": "audit",
            "target_agent": "detail_augmentation" if not approved else "__end__",
        }