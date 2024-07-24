import numpy as np
import matplotlib.pyplot as plt

# 生成有用信号和噪声
def generate_signals(num_samples, noise_level):
    # 有用信号：正弦波
    t = np.arange(num_samples)
    s = np.sin(0.1 * t)  # 有用信号
    # 噪声：高斯白噪声
    v = noise_level * np.random.randn(num_samples)  # 噪声
    # 输入信号
    d = s + v
    return s, v, d

# LMS算法实现
def lms_filter(d, x, mu, num_taps):
    num_samples = len(d)
    w = np.zeros(num_taps)  # 初始化滤波器系数
    y = np.zeros(num_samples)  # 输出信号
    e = np.zeros(num_samples)  # 误差信号

    for n in range(num_taps, num_samples):
        # 生成输入向量
        x_n = x[n:n - num_taps:-1]  # 参考信号的倒序
        y[n] = np.dot(w, x_n)  # 计算输出信号
        e[n] = d[n] - y[n]  # 计算误差信号
        w += mu * e[n] * x_n  # 更新滤波器系数

    return y, e, w
# 参数设置
num_samples = 2000
noise_level = 0.5
mu = 0.01  # 学习率
num_taps = 32  # 滤波器阶数

# 生成信号
s, v, d = generate_signals(num_samples, noise_level)

# 生成参考噪声信号（与噪声相关）
x = v + 0.05 * np.random.randn(num_samples)  # 添加一些小的随机干扰

# 应用LMS自适应滤波器
y, e, w = lms_filter(d, x, mu, num_taps)

# 绘制结果
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(d, label='Input Signal (d[n])')
plt.plot(s, label='Desired Signal (s[n])', linestyle='--')
plt.title('Input Signal and Desired Signal')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(y, label='Output Signal (y[n])', color='orange')
plt.title('Output Signal from Adaptive Filter')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(e, label='Error Signal (e[n])', color='red')
plt.title('Error Signal (Estimated Desired Signal)')
plt.legend()

plt.tight_layout()
plt.show()