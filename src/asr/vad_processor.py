from silero_vad import load_silero_vad
from config.settings import config
import logging
import numpy as np
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VadCfg:
    sr: int = 16000            # 采样率（必须与采集一致）
    block: int = 512           # 每帧样本数（你现在就是512）
    pre_roll_ms: int = 200     # 段开始前附带的音频（前滚）
    post_roll_ms: int = 200    # 段结束后附带的音频（后滚）
    start_frames: int = 3      # 连续有声帧数≥ 才算“开始”（迟滞）
    end_frames: int = 6        # 连续静音帧数≥ 才算“结束”（迟滞）
    min_dur_ms: int = 250      # 短段过滤阈值（给外层参考，可不强制）
    thr_db_above: float = 8.0  # 相对噪声底的能量门限（dB）
    noise_ewma: float = 0.995  # 噪声底跟踪平滑系数（越接近1越平滑）
    clamp_db_floor: float = -60.0  # 能量下限（防止log过小）
    silero_prob_thr: float = 0.5   # Silero 概率阈值（与能量双门限 AND）


class VADProcessor:
    """VAD处理器（Silero 概率 + 能量双门限，带迟滞/前后滚）"""

    def __init__(self, cfg: VadCfg | None = None):
        self.cfg = cfg or VadCfg(sr=config.SAMPLE_RATE, block=512)
        self.vad = None
        self._initialize_vad()

        # 状态
        self.frame_idx = 0
        self.in_speech = False
        self.pos_frames = 0
        self.neg_frames = 0
        self.noise_db = -50.0
        self.post_hang = 0  # 计数用于后滚

        # 预先算好帧到毫秒的换算
        self.ms_per_frame = 1000.0 * self.cfg.block / self.cfg.sr
        self.pre_frames = max(0, int(round(self.cfg.pre_roll_ms / self.ms_per_frame)))
        self.post_frames = max(0, int(round(self.cfg.post_roll_ms / self.ms_per_frame)))

        # 可选：记录最近一次“开始”帧索引给外层用（比如做最小时长判断/合并）
        self._seg_start_idx = None

    def _initialize_vad(self):
        """初始化 Silero VAD 模型"""
        try:
            self.vad = load_silero_vad()  # 返回 torch 模型
            self.vad.eval()
            logger.info("VAD模型加载成功")
        except Exception as e:
            logger.error(f"VAD模型加载失败: {e}")
            raise

    # —— 工具：计算一帧的RMS dB（单声道） ——
    def _frame_db(self, x: np.ndarray) -> float:
        if x.ndim == 2:
            x = x[:, 0]
        rms = np.sqrt(np.mean(x.astype(np.float32) ** 2) + 1e-12)
        db = 20.0 * np.log10(max(rms, 1e-6))
        return max(db, self.cfg.clamp_db_floor)

    # —— 工具：用 Silero 得到该帧“有声”概率 ——
    def _silero_prob(self, x: np.ndarray) -> float:
        # Silero 需要 1D float32，长度为当前块；采样率传 sr
        if x.ndim == 2:
            x = x[:, 0]
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32))
            # 部分 silero 实现是 (waveform, sr) 或 (waveform, sample_rate=)
            # 这里按常见签名：vad(waveform, sr)
            p = self.vad(t, self.cfg.sr)
            # 返回标量概率
            if isinstance(p, (list, tuple)):
                p = p[0]
            if isinstance(p, torch.Tensor):
                p = p.item()
            return float(p)

    def process_samples(self, samples: np.ndarray):
        """
        输入：一帧样本 (block, 1或block,)
        输出：None / {'start': True} / {'end': True}
        """
        # 递增帧号
        self.frame_idx += 1

        # 计算本帧能量与 Silero 概率
        db = self._frame_db(samples)
        prob = self._silero_prob(samples)

        # 自适应噪声底（仅在“非说话状态”下跟踪，避免被语音抬高）
        if not self.in_speech:
            self.noise_db = self.cfg.noise_ewma * self.noise_db + (1.0 - self.cfg.noise_ewma) * db

        # 双门限：能量相对噪声底 + Silero 概率
        energy_ok = (db >= self.noise_db + self.cfg.thr_db_above)
        silero_ok = (prob >= self.cfg.silero_prob_thr)
        voiced = energy_ok and silero_ok

        # 迟滞计数
        if voiced:
            self.pos_frames += 1
            self.neg_frames = 0
        else:
            self.neg_frames += 1
            self.pos_frames = 0

        # —— 进入语音（带前滚） ——
        if not self.in_speech and self.pos_frames >= self.cfg.start_frames:
            self.in_speech = True
            # 记录起点索引（向前带 pre_frames）
            self._seg_start_idx = max(0, self.frame_idx - self.pos_frames - self.pre_frames)
            self.post_hang = 0
            return {'start': True}

        # —— 退出语音（带后滚） ——
        if self.in_speech:
            if self.neg_frames >= self.cfg.end_frames:
                self.post_hang += 1
                if self.post_hang >= self.post_frames:
                    self.in_speech = False
                    # 段结束：真正触发 end（你外层会用 prev_audio_chunk 做前滚、post_hang 也已经延后）
                    return {'end': True}

        return None
