"""AgentForNovel —— 基于 LangGraph 的小说半自动化创作系统（最小化运行）

流程: detail -> dialogue，仅经过 detail 和 dialogue 两个 agent。
从 example/input.txt 读取全文作为小说内容，输出到 example/output.txt。
"""

from dotenv import load_dotenv

from src.graph import create_novel_writing_graph

load_dotenv()

INPUT_FILE = "example/input.txt"
OUTPUT_FILE = "example/output.txt"


def main():
    """主入口：从文件读取输入，运行小说写作流程，输出到文件"""
    # 读取整个输入文件作为小说内容
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        novel_content = f.read().strip()

    print(f"读取输入: {INPUT_FILE} ({len(novel_content)} 字符)")
    print("\n开始小说写作流程: detail -> dialogue\n")

    graph = create_novel_writing_graph()
    result = graph.run(novel_content=novel_content)

    # 写入输出文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(result["novel_content"])

    print("=" * 60)
    print("创作完成！")
    print(f"输出已写入: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()