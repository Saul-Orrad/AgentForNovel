"""异步保存工具 —— 将内容写入 txt 文件，并在 SQLite 中记录元数据"""

import asyncio
import sqlite3
import time
from datetime import datetime
from pathlib import Path


class AsyncSaveTool:
    """异步保存工具

    将 content 保存为 txt 文件到 output/YYYYMMDD/ 目录下，
    文件名为 {stateId}_{agentId}_{timestamp}.txt，
    并将 stateId、agentId、timestamp 记录到 SQLite 数据库。
    """

    def __init__(self, base_dir: str | None = None):
        if base_dir is None:
            base_dir = str(Path(__file__).parent.parent.parent / "output")
        self._base_dir = Path(base_dir)
        self._db_path = self._base_dir / "records.db"
        self._init_db()

    def _init_db(self) -> None:
        """初始化 SQLite 数据库和表结构"""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS save_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                file_path TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    async def save(self, state_id: str, content: str, agent_id: str) -> str:
        """异步保存内容

        Args:
            state_id: 状态 ID
            content:  要保存的文本内容
            agent_id: Agent ID

        Returns:
            str: 保存的 txt 文件路径
        """
        timestamp = int(time.time())
        date_dir = datetime.now().strftime("%Y%m%d")
        filename = f"{state_id}_{agent_id}_{timestamp}.txt"
        dir_path = self._base_dir / date_dir
        file_path = dir_path / filename

        # 异步写入文件
        await asyncio.to_thread(self._write_file, dir_path, file_path, content)

        # 异步写入数据库
        await asyncio.to_thread(
            self._insert_record, state_id, agent_id, timestamp, str(file_path)
        )

        return str(file_path)

    @staticmethod
    def _write_file(dir_path: Path, file_path: Path, content: str) -> None:
        """同步写入 txt 文件"""
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _insert_record(
        self, state_id: str, agent_id: str, timestamp: int, file_path: str
    ) -> None:
        """同步写入 SQLite 记录"""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            "INSERT INTO save_records (state_id, agent_id, timestamp, file_path) VALUES (?, ?, ?, ?)",
            (state_id, agent_id, timestamp, file_path),
        )
        conn.commit()
        conn.close()

    def list_records(self, state_id: str | None = None) -> list[dict]:
        """查询保存记录

        Args:
            state_id: 可选，按 state_id 过滤

        Returns:
            list[dict]: 记录列表
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        if state_id:
            rows = conn.execute(
                "SELECT * FROM save_records WHERE state_id = ? ORDER BY timestamp DESC",
                (state_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM save_records ORDER BY timestamp DESC"
            ).fetchall()
        conn.close()
        return [dict(row) for row in rows]