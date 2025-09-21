import os
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_jsonl_file(directory: str = None) -> str:
    try:
        if directory is None:
            directory = os.path.join(BASE_DIR, "data", "outputs", "json")

        if not os.path.exists(directory):
            return ""

        jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))

        if not jsonl_files:
            return ""

        latest_file = max(jsonl_files, key=os.path.getmtime)
        return latest_file

    except Exception as e:
        print(f"查找JSONL文件失败: {e}")
        return ""

print(find_jsonl_file())
print(BASE_DIR)