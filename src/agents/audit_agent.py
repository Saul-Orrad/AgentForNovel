import json
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import create_audit_llm


class AuditAgent:

    def __init__(self, llm: BaseChatModel | None = None):
        self._llm = llm or create_audit_llm()
        self._system_prompt = self._load_prompt()

    @staticmethod
    def _load_prompt() -> str:
        prompt_path = Path(__file__).parent.parent.parent / "config" / "prompt" / "audit_prompt.yml"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return ""

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def invoke(self, state: dict) -> dict:
        messages = [SystemMessage(content=self._system_prompt)]
        user_message = state.get("novel_content", "")
        messages.append(HumanMessage(content=user_message))

        response = self._llm.invoke(messages)
        parsed = self._parse_response(response.content)

        audit_result = parsed.get("audit_result", 1)
        audit_feedback = parsed.get("audit_feedback", "")

        return {
            "audit_result": audit_result,
            "audit_feedback": audit_feedback,
            "source_agent": "audit",
            "target_agent": "detail_augmentation" if audit_result != 0 else "__end__",
        }

    @staticmethod
    def _parse_response(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if len(lines) > 2 else lines)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}