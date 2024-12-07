import torch

# 加载语音识别模型
model, decoder, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_stt',
    language='en'  # 可选 'de'（德语）、'es'（西班牙语）等
)

# 读取音频文件并进行识别
audio_path = 'audio.wav'
audio = utils.read_audio(audio_path)
text = model(audio)
print(decoder(text[0]))
