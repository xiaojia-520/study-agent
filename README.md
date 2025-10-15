# 🧠 Study Agent

面向课堂或会议场景的 **Study Agent** 是一套将 **实时语音转录（ASR）**、**结构化笔记** 与 **检索增强问答（RAG）** 整合在一起的多模态学习助手。系统会监听语音、进行端点检测、转写与自动标点，并把文本增量写入 JSONL 与向量数据库，问答时通过 RAG 召回最相关上下文。

---

## ✨ 核心功能
- **实时录音与端点检测**：`sounddevice` 采集音频 + 自研 `VADProcessor` 区分语音段/静音段  
- **高质量语音识别**：基于 FunASR 的 `speech_paraformer-large-vad-punc` 中文 ASR，支持按课程热词  
- **自动标点与文本清洗**：`PuncProcessor` 补标点，问答前由 `RAGProcessor` 规整口语化文本  
- **结构化笔记**：每段语音写入 `data/outputs/json/*.jsonl`，包含起止时间、时长、文本等  
- **向量化与语义检索**：`bge-small-zh-v1.5` 生成向量，经 `EmbeddingManager` 写入 Qdrant  
- **检索增强问答（RAG）**：从 JSONL + 向量库召回上下文，调用 LLM 生成结构化答案  
- **可视化 Web Demo**：`web_demo/app.py`（Flask）用于浏览器端录制/查看转写与历史问答

---

## 🗂️ 目录速览
study-agent/
├─ main.py # 命令行入口：ASR + Embedding + RAG
├─ setup_model.bat # （可选）模型准备脚本
├─ config/
│ ├─ settings.py # 全局配置（采样率、Qdrant、设备等）
│ ├─ prompts.py # LLM 提示词模板
│ ├─ .env.example # 环境变量示例（请复制为 .env 并填写）
├─ src/
│ ├─ asr/ # 录音、VAD、ASR、标点
│ ├─ embedding/ # 向量生成与 Qdrant 客户端
│ ├─ llm/ # 模型封装与 RAG 逻辑
│ ├─ models/ # 数据结构/Schema
│ └─ utils/ # 日志/文件/时间等工具
├─ web_demo/ # Flask Web 前端
├─ data/
│ ├─ models/ # 模型目录（.gitkeep 占位，不上传权重）
│ ├─ outputs/json/ # 转写结果（JSONL）
│ └─ logs/ # 运行日志
├─ requirements.txt
└─ tools.py

---

## 🚀 快速开始

### 1) 安装依赖（Python 3.10+）
```bash
pip install -r requirements.txt

2) 配置环境变量

复制示例并填写你的密钥等：
cp config/.env.example config/.env

.env 中常见变量：

OPENAI_API_KEY=你的API密钥
QDRANT_HOST=localhost
QDRANT_PORT=6333


本项目使用 python-dotenv 自动读取 config/.env；.env 已在 .gitignore 中忽略，不会被上传。

3) 准备模型

方式 A（推荐）：执行根目录下脚本（若已提供）

./setup_model.bat


方式 B：手动放置至（示例）：

data/models/asr/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch

data/models/punc/punc_ct-transformer_cn-en-common-vocab471067-large

data/models/embedding/bge-small-zh-v1.5

4) 启动 Qdrant（向量数据库）
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant


确保 .env 或 config/settings.py 中的 QDRANT_HOST/PORT 与之匹配。

5) 命令行运行
python main.py --mode both --lesson 工程热力学


--mode asr：仅实时录音+转写

--mode qa：仅对历史 JSONL 做问答

--mode both：边录边转写并进入问答

--lesson：课程/会议名称（写入 JSONL & 匹配热词）

转写将增量写入：data/outputs/json/<timestamp>.jsonl
日志保存在：data/logs/

6) 启动 Web Demo（可选）

Linux/macOS：

export FLASK_APP=web_demo.app
flask run --host 0.0.0.0 --port 5000


Windows PowerShell：

$env:FLASK_APP="web_demo.app"
flask run --host 0.0.0.0 --port 5000


访问 http://localhost:5000 使用图形界面录制/查看。

⚙️ 配置说明

采样与设备：在 config/settings.py 调整 SAMPLE_RATE、DEVICE 等

热词：在 HOTWORDS 中为不同课程/会议配置专属词表

向量化：BATCH_SIZE 决定批量提交大小（权衡延迟/吞吐）

日志：默认 INFO，可设 LOG_LEVEL 或 LOG_FILE

🧪 开发与调试

先用短音频自测模型加载：python test/test.py

可在 src/llm/model_manager.py 接入自定义 LLM（OpenAI / Ollama / 本地模型等）

建议按课程管理 session，方便区分 JSONL 与向量库数据