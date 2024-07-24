from IPython.display import Audio

# 播放音频文件
def play_audio(file_path):
    return Audio(file_path)

# 替换为你的音频文件路径
audio = play_audio('D:\Add_voice.wav')  # 或 'output_audio.wav'
audio