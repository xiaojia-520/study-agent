# 采集进程 ---------------------------------------------------------
import numpy as np, sounddevice as sd
from multiprocessing import shared_memory, Value
import ctypes, threading

FS = 16000
CAP_S = 60
CAP = FS * CAP_S  # 60 秒缓冲
BLOCK = 1024  # 回调块

shm = shared_memory.SharedMemory(create=True, size=CAP * 2)  # int16 -> 2 bytes
buf = np.ndarray((CAP,), dtype=np.int16, buffer=shm.buf)

wptr = Value(ctypes.c_longlong, 0)  # 写指针（样本索引，单调递增）


def audio_cb(indata, frames, time, status):
    if status:
        # 丢到日志
        pass
    x = np.frombuffer(indata, dtype=np.int16)[:, 0]  # 单声道视图
    n = x.shape[0]
    wp = wptr.value % CAP
    tail = CAP - wp
    if n <= tail:
        buf[wp:wp + n] = x
    else:
        buf[wp:] = x[:tail]
        buf[:n - tail] = x[tail:]
    wptr.value += n


with sd.RawInputStream(samplerate=FS, channels=1, dtype='int16',
                       blocksize=BLOCK, callback=audio_cb):
    threading.Event().wait()  # keep alive
