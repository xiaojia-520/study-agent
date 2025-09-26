#!/usr/bin/env python3
"""
主入口文件 - 实时语音转录与智能问答系统
"""

import sys
import os
import time

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.asr.recorder import AudioRecorder
from src.asr.vad_processor import VADProcessor
from src.asr.asr_processor import ASRProcessor
from src.asr.punc_processor import PuncProcessor
from src.embedding.embedding_manager import EmbeddingManager
from src.llm.rag_processor import RAGProcessor
from src.utils.logger import setup_logging
from src.utils.file_utils import ensure_directory
import argparse
from threading import Thread
from src.utils.file_utils import find_jsonl_file


def main():
    """主函数"""
    # 初始化日志
    logger = setup_logging()
    logger.info("启动实时语音转录与智能问答系统")

    print("=" * 50)
    print("🎤 实时语音转录与智能问答系统")
    print("=" * 50)

    # 确保输出目录存在
    ensure_directory('data/outputs/json')
    ensure_directory('data/logs')

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='实时语音转录与智能问答系统')
    parser.add_argument('--mode', choices=['asr', 'qa', 'both'], default='both',
                        help='运行模式: asr(仅语音识别), qa(仅问答), both(两者)')
    parser.add_argument('--lesson', type=str, help='课程名称')
    args = parser.parse_args()

    # 获取课程名称
    lesson_name = args.lesson or input("请输入课程名称: ").strip()

    try:
        # 初始化组件
        vad_processor = VADProcessor()
        asr_processor = ASRProcessor()
        punc_processor = PuncProcessor()
        with EmbeddingManager() as embedding_manager:

            # 设置ASR处理器的标点处理器
            asr_processor.punc_processor = punc_processor
            recording_started_at = None
            if args.mode in ['asr', 'both']:
                recording_started_at = time.time()
                logger.info(f"开始录制课程: {lesson_name}")
                recorder = AudioRecorder()
                asr_thread = Thread(
                    target=recorder.start_recording,
                    kwargs=dict(
                        vad_processor=vad_processor,
                        asr_processor=asr_processor,
                        embedding_manager=embedding_manager,
                        lesson_name=lesson_name
                    ),
                    daemon=(args.mode != 'asr')
                )
                asr_thread.start()

                if args.mode == 'asr':
                    try:
                        asr_thread.join()
                    except KeyboardInterrupt:
                        logger.info("用户中断录音")
                    finally:
                        recorder.stop_recording()
                        asr_thread.join(timeout=2)
                        return
            if args.mode in ['qa', 'both']:
                rag_processor = RAGProcessor()
                wait_for_new_session = args.mode == 'both' and recording_started_at is not None
                jsonl_path = ""
                notified_waiting = False

                while True:
                    candidate = find_jsonl_file()
                    if candidate:
                        if not wait_for_new_session or os.path.getmtime(candidate) >= recording_started_at:
                            jsonl_path = candidate
                            if wait_for_new_session and notified_waiting:
                                logger.info(f"检测到新的转录文件: {jsonl_path}")
                            break

                    if not wait_for_new_session:
                        break

                    if not notified_waiting:
                        logger.info("等待新的转录文件生成...")
                        notified_waiting = True

                    time.sleep(1)
                if jsonl_path:
                    logger.info(f"使用转录文件: {jsonl_path}")

                    # 问答循环
                    while True:
                        try:
                            question = input("\n请输入问题 (输入 'quit' 退出): ")
                            if question.lower() in ['quit', 'exit', 'q']:
                                break

                            response = rag_processor.generate_response(question, jsonl_path, session_id=lesson_name)
                            print(f"\n回答: {response}")

                        except KeyboardInterrupt:
                            logger.info("用户中断问答")
                            break
                        except Exception as e:
                            logger.error(f"问答过程出错: {e}")
                else:
                    logger.warning("未找到转录文件，无法进行问答")

    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
    finally:
        recorder.stop_recording()
        logger.info("程序结束")


if __name__ == "__main__":
    main()
