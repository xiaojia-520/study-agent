# Study Agent

面向课堂或会议场景的“Study Agent”是一套将**实时语音转录**、**知识整理**和**检索式问答**整合在一起的多模态学习助手。系统会自动监听语音、分段、转写、打标点，并将文本增量写入 JSONL 与向量数据库，最终通过 RAG（Retrieval-Augmented Generation）在问答时召回最相关的上下文。

## ✨ 核心功能
- **实时录音与端点检测**：使用 `sounddevice` 采集音频，结合自定义的 `VADProcessor` 自动区分说话段与静音段。
- **高质量语音识别**：基于 FunASR 的 `speech_paraformer-large-vad-punc` 模型完成转写，可按课程名称加载热词提升识别率。
- **自动标点与文本清洗**：`PuncProcessor` 对识别文本补齐标点，`RAGProcessor` 在问答前再次规整口语化填充词。
- **结构化笔记落地**：每个语音片段被写入 `data/outputs/json/*.jsonl`，记录开始/结束时间、时长、文本等信息。
- **向量化与知识库同步**：`EmbeddingManager` 按批处理转写内容，调用 `bge-small-zh-v1.5` 生成向量并存入 Qdrant，支持后续语义检索。
- **检索增强问答**：`RAGProcessor` 同时读取实时 JSONL 与向量库召回的上下文，通过大模型生成结构化答案。
- **可视化 Web Demo**：`web_demo/app.py` 提供 Flask 前端，便于在浏览器中启动/停止录制并查看转写结果。

## 🗂️ 目录速览
```
study-agent/
├─ main.py                 # 命令行入口，串联录音、ASR、嵌入、RAG
├─ config/
│  ├─ settings.py          # 采样率、模型路径、Qdrant 等全局配置
│  └─ prompts.py           # LLM 提示词模板
├─ src/
│  ├─ asr/                 # 录音、VAD、ASR、标点等语音处理模块
│  ├─ embedding/           # 嵌入模型与 Qdrant 管理
│  ├─ llm/                 # 大模型封装与 RAG 逻辑
│  ├─ models/              # 业务数据模型
│  └─ utils/               # 日志、文件、时间等工具
├─ web_demo/               # Flask Web 端示例
├─ requirements.txt
└─ data/                   # 建议放置模型、输出与日志
```

## 🚀 快速开始
### 1. 准备运行环境
1. 安装 Python 3.10+。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 下载模型（可参考 FunASR 与 HuggingFace Hub），并放置到默认目录：
   - `data/models/asr/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
   - `data/models/punc/punc_ct-transformer_cn-en-common-vocab471067-large`
   - `data/models/embedding/bge-small-zh-v1.5`

### 2. 启动 Qdrant 向量数据库
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```
确保 `config/settings.py` 中的 `QDRANT_HOST` 与 `QDRANT_PORT` 指向可访问的实例。

### 3. 运行命令行流程
```bash
python main.py --mode both --lesson 工程热力学
```
- `--mode asr`：只做实时录音+转写。
- `--mode qa`：仅加载已有 JSONL 做问答。
- `--mode both`：边录边写，并等待最新的转写结果后进入问答循环。
- `--lesson`：课程/会议名称，会写入 JSONL 并用于热词配置。

转写结果会以增量写入 `data/outputs/json/<timestamp>.jsonl`，日志默认保存在 `data/logs/`。

### 4. 可选：启动 Web Demo
```bash
export FLASK_APP=web_demo.app
flask run --host 0.0.0.0 --port 5000
```
浏览器访问 `http://localhost:5000`，即可通过界面启动录制、查看实时转写与历史问答。

## ⚙️ 配置说明
- 若有自定义麦克风或声卡，可在 `config/settings.py` 中调整 `DEVICE`、`SAMPLE_RATE` 等参数。
- `HOTWORDS` 字典用于针对不同课程启用专属热词；可按需扩展。
- `BATCH_SIZE` 控制向量化批量提交的大小；过小会频繁写入，过大会增加延迟。
- 默认日志级别为 `INFO`，可修改 `LOG_LEVEL` 或设置 `LOG_FILE` 输出路径。

## 🧪 开发与调试建议
- 在正式场景前使用短音频文件验证模型是否正确加载：`python test/test.py`。
- 如果需要替换 LLM，可在 `src/llm/model_manager.py` 中扩展自定义模型（例如 OpenAI、Ollama、本地大模型）。
- 建议为不同课程创建独立的 session，便于区分 JSONL 与 Qdrant 记录。

## 📄 License
本项目代码仅用于个人学习与实验，暂未指定开源协议，请在遵守相应模型与数据授权的前提下使用。