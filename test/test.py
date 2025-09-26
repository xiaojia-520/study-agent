from funasr import AutoModel

model = AutoModel(
    model=r"D:\python-work\study-agent-main\data\models\asr\speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)

r = model.inference(input=r"D:\python-work\study-agent-main\data\models\asr\SenseVoiceSmall\example\zh.mp3")

print(r)