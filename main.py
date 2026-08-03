"""AgentForNovel —— 基于 LangGraph 的小说半自动化创作系统

流程: detail -> dialogue -> audit，audit 未通过则回到 detail 重试，
detail 和 dialogue 需要时可调用 extract 子节点提取关键信息。
"""

import os
from dotenv import load_dotenv

from src.graph import create_novel_writing_graph

load_dotenv()


def main():
    """主入口：运行小说写作流程"""
    novel_content = input("请输入初始小说内容（可为空）: ").strip()
    dynamic_prompt = input("请输入提取提示词（可为空）: ").strip()

    graph = create_novel_writing_graph()
    print("\n开始小说写作流程: detail -> dialogue -> audit\n")

    result = graph.run(novel_content=novel_content, dynamic_prompt=dynamic_prompt)

    print("\n" + "=" * 60)
    print("创作完成！")
    print(f"审核结果: {'通过' if result['audit_result'] else '未通过'}")
    print("=" * 60)
    print("\n--- 最终小说内容 ---\n")
    print(result["novel_content"])
    print("\n--- 审核反馈 ---\n")
    print(result["audit_feedback"])
    if result.get("extract_result"):
        print("\n--- 提取结果 ---\n")
        print(result["extract_result"])


if __name__ == "__main__":
    main()