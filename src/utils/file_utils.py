import os
import json
from pathlib import Path
from typing import List, Any, Optional
import datetime
from uuid import uuid5, NAMESPACE_DNS


def get_parent_dir(path, n=1):
    for _ in range(n):
        path = os.path.dirname(path)
    return path


BASE_DIR = get_parent_dir(os.path.abspath(__file__), 3)


def ensure_directory(directory_path: str) -> None:
    """确保目录存在，如果不存在则创建"""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def get_next_id(log_file: str) -> int:
    """获取下一个ID"""
    try:
        with open(log_file, "r", encoding="utf-8") as fr:
            return sum(1 for _ in fr) + 1
    except FileNotFoundError:
        return 1


def load_json(json_file: str) -> List[Any]:
    """加载JSON文件"""
    if not os.path.exists(json_file):
        return []

    items = []
    try:
        with open(json_file, "r", encoding="utf-8") as fr:
            lines = fr.readlines()

        tail = lines[-10:]  # 只读取最后10行
        for line in tail:
            try:
                payload = json.loads(line)
                id_val = payload.get("id", 1)
                session_id = payload.get("session_id", "default")
                pid = str(uuid5(NAMESPACE_DNS, f"{session_id}-{id_val}"))
                text = payload.get("text", "")
                items.append((text, payload, pid))
            except Exception:
                continue
    except Exception as e:
        print(f"加载JSON文件失败: {e}")

    return items


def write_jsonl(file_path: str, data: dict) -> None:
    """写入JSONL文件"""
    ensure_directory(Path(file_path).parent)
    with open(file_path, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


def _iter_unique_dirs(paths):
    seen = set()
    for path in paths:
        try:
            resolved = Path(path)
        except TypeError:
            continue
        try:
            key = resolved.resolve()
        except OSError:
            key = resolved
        if key in seen:
            continue
        seen.add(key)
        yield resolved


def find_jsonl_file(directory: str = None) -> str:
    try:
        candidate_dirs = []
        if directory is not None:
            candidate_dirs.append(directory)
        else:
            base = Path(BASE_DIR)
            candidate_dirs.extend(
                [
                    base / "data" / "outputs" / "json",
                    base / "web_demo" / "data" / "outputs" / "json",
                    Path.cwd() / "data" / "outputs" / "json",
                ]
            )

        latest_path: Optional[Path] = None
        latest_mtime = -1.0

        for folder in _iter_unique_dirs(candidate_dirs):
            if not folder.exists():
                continue
            try:
                files = list(folder.glob("*.jsonl"))
            except OSError:
                continue
            for jsonl_file in files:
                try:
                    mtime = jsonl_file.stat().st_mtime
                except OSError:
                    continue
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_path = jsonl_file
        return str(latest_path) if latest_path else ""
    except Exception as e:
            print(f"查找JSONL文件失败: {e}")
            return ""
