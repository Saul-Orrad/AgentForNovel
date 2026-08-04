from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.detail_augmentation_agent import DetailAugmentationAgent
from src.agents.dialogue_complementation_agent import DialogueComplementationAgent
from .novel_state import NovelWritingState


class NovelWritingGraph:
    """小说写作图（最小化运行） —— detail -> dialogue"""

    def __init__(
        self,
        detail_agent: DetailAugmentationAgent | None = None,
        dialogue_agent: DialogueComplementationAgent | None = None,
    ):
        self.detail_agent = detail_agent or DetailAugmentationAgent()
        self.dialogue_agent = dialogue_agent or DialogueComplementationAgent()
        self._graph = self._build_graph()

    def _detail_node(self, state: NovelWritingState) -> dict:
        result = self.detail_agent.invoke(state)
        result["history"] = [{"role": "detail_augmentation"}]
        return result

    def _dialogue_node(self, state: NovelWritingState) -> dict:
        result = self.dialogue_agent.invoke(state)
        result["history"] = [{"role": "dialogue_complementation"}]
        return result

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(NovelWritingState)

        builder.add_node("detail_augmentation", self._detail_node)
        builder.add_node("dialogue_complementation", self._dialogue_node)

        # 流程: detail -> dialogue -> end
        builder.set_entry_point("detail_augmentation")
        builder.add_edge("detail_augmentation", "dialogue_complementation")
        builder.add_edge("dialogue_complementation", END)

        memory = MemorySaver()
        return builder.compile(checkpointer=memory)

    @property
    def graph(self):
        return self._graph

    def run(self, novel_content: str = "", dynamic_prompt: str = "", config: dict | None = None) -> NovelWritingState:
        initial_state: NovelWritingState = {
            "source_agent": "",
            "target_agent": "detail_augmentation",
            "novel_content": novel_content,
            "dynamic_prompt": dynamic_prompt,
            "extract_result": "",
            "history": [],
        }

        if config is None:
            config = {"configurable": {"thread_id": "default"}}

        return self._graph.invoke(initial_state, config)


def create_novel_writing_graph(
    detail_agent: DetailAugmentationAgent | None = None,
    dialogue_agent: DialogueComplementationAgent | None = None,
) -> NovelWritingGraph:
    return NovelWritingGraph(
        detail_agent=detail_agent,
        dialogue_agent=dialogue_agent,
    )