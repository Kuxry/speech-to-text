import torch
import numpy as np
import sounddevice as sd
from queue import Queue
from threading import Thread

# 加载 Silero 模型
model, decoder, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_stt',
    language='en',  # 替换为需要的语言，例如 'en', 'de', 'es' 等
    trust_repo=True
)
(read_audio, prepare_model_input, *_ ) = utils

# 参数设置
SAMPLE_RATE = 16000  # Silero 模型采样率
CHUNK_SIZE = 512     # 缩短音频块大小，约 0.032 秒
BUFFER_SIZE = SAMPLE_RATE  # 缓冲区长度

# 音频队列，用于存储实时音频流
audio_queue = Queue()

# 模型推理线程
def inference_worker():
    buffer = np.zeros(0, dtype=np.float32)  # 用于存储连续音频数据
    print("推理线程已启动...")
    while True:
        if not audio_queue.empty():
            # 从队列中取出音频块
            audio_data = audio_queue.get()
            audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
            buffer = np.append(buffer, audio_chunk)  # 加入缓冲区

            # 如果缓冲区足够大，进行推理
            if len(buffer) >= SAMPLE_RATE // 2:  # 0.5 秒的音频块
                prepared_input = prepare_model_input(buffer[:SAMPLE_RATE // 2], SAMPLE_RATE)
                if isinstance(prepared_input, list):
                    prepared_input = np.array(prepared_input)
                input_batch = torch.tensor(prepared_input)
                buffer = buffer[SAMPLE_RATE // 2:]  # 保留剩余音频数据

                # 模型推理
                output = model(input_batch)
                decoded_text = decoder(output[0])
                if decoded_text.strip():
                    print(f"识别结果：{decoded_text}")

# 音频回调函数
def audio_callback(indata, frames, time, status):
    # 将音频块写入队列
    audio_queue.put(indata.copy())

# 主函数：启动音频流和推理线程
def main():
    print("开始实时语音识别，按 Ctrl+C 退出...")
    inference_thread = Thread(target=inference_worker, daemon=True)
    inference_thread.start()  # 启动推理线程

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            while True:
                pass  # 主线程保持运行
    except KeyboardInterrupt:
        print("实时语音识别结束。")

# 启动程序
if __name__ == "__main__":
    main()
