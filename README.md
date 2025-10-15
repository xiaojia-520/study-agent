# \#Study Agent

# 

# 面向课堂或会议场景的“Study Agent”是一套将\*\*实时语音转录\*\*、\*\*知识整理\*\*和\*\*检索式问答\*\*整合在一起的多模态学习助手。系统会自动监听语音、分段、转写、打标点，并将文本增量写入 JSONL 与向量数据库，最终通过 RAG（Retrieval-Augmented Generation）在问答时召回最相关的上下文。

# 

# \## ✨ 核心功能

# \- \*\*实时录音与端点检测\*\*：使用 `sounddevice` 采集音频，结合自定义的 `VADProcessor` 自动区分说话段与静音段。

# \- \*\*高质量语音识别\*\*：基于 FunASR 的 `speech\_paraformer-large-vad-punc` 模型完成转写，可按课程名称加载热词提升识别率。

# \- \*\*自动标点与文本清洗\*\*：`PuncProcessor` 对识别文本补齐标点，`RAGProcessor` 在问答前再次规整口语化填充词。

# \- \*\*结构化笔记落地\*\*：每个语音片段被写入 `data/outputs/json/\*.jsonl`，记录开始/结束时间、时长、文本等信息。

# \- \*\*向量化与知识库同步\*\*：`EmbeddingManager` 按批处理转写内容，调用 `bge-small-zh-v1.5` 生成向量并存入 Qdrant，支持后续语义检索。

# \- \*\*检索增强问答\*\*：`RAGProcessor` 同时读取实时 JSONL 与向量库召回的上下文，通过大模型生成结构化答案。

# \- \*\*可视化 Web Demo\*\*：`web\_demo/app.py` 提供 Flask 前端，便于在浏览器中启动/停止录制并查看转写结果。

# 

# \## 🗂️ 目录速览

# ```

# study-agent/

# ├─ main.py                 # 命令行入口，串联录音、ASR、嵌入、RAG

# ├─ config/

# │  ├─ settings.py          # 采样率、模型路径、Qdrant 等全局配置

# │  └─ prompts.py           # LLM 提示词模板

# ├─ src/

# │  ├─ asr/                 # 录音、VAD、ASR、标点等语音处理模块

# │  ├─ embedding/           # 嵌入模型与 Qdrant 管理

# │  ├─ llm/                 # 大模型封装与 RAG 逻辑

# │  ├─ models/              # 业务数据模型

# │  └─ utils/               # 日志、文件、时间等工具

# ├─ web\_demo/               # Flask Web 端示例

# ├─ requirements.txt

# └─ data/                   # 建议放置模型、输出与日志

# ```

# 

# \## 🚀 快速开始

# \### 1. 准备运行环境

# 1\. 安装 Python 3.10+。

# 2\. 安装依赖：

# &nbsp;  ```bash

# &nbsp;  pip install -r requirements.txt

# &nbsp;  ```

# 3\. 下载模型：

# 运行setup\_model.bat

# 

# \### 2. 启动 Qdrant 向量数据库

# ```bash

# docker run -p 6333:6333 -v $(pwd)/qdrant\_storage:/qdrant/storage qdrant/qdrant

# ```

# 

# \### 3. 运行命令行流程

# 在web\_demo下运行

# ```bash

# python app.py

# ```

# 

# \## ⚙️ 配置说明

# \- 在 `config/settings.py` 中调整 `DEVICE`、`SAMPLE\_RATE` 等参数。(主要是device 用这个命令看一下你用的设备是哪个 print(sd.query\_devices()))

# \- `HOTWORDS` 字典用于针对不同课程启用专属热词；可按需扩展。

# \- `BATCH\_SIZE` 控制向量化批量提交的大小；过小会频繁写入，过大会增加延迟。

# \- 默认日志级别为 `INFO`，可修改 `LOG\_LEVEL` 或设置 `LOG\_FILE` 输出路径。



