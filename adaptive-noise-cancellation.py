import numpy as np
from scipy.io import loadmat, wavfile
import matplotlib.pyplot as plt
import simpleaudio as sa

# Load the MAT file
mat_data = loadmat(r'D:\Zhou_results.mat')

# Extract sample rate and audio data
sample_rate = int(mat_data['Fs'][0][0])  # Sample rate
data = mat_data['y']

# If the data has multiple channels, select one channel (e.g., left channel)
if data.ndim > 1:
    data = data[:, 0]

# Select the first 10 seconds of audio data (assuming a sample rate of 44100 Hz)
duration = 10  # seconds
num_samples = sample_rate * duration
data = data[:num_samples]

# Ensure the data is of float type and normalize to [-1, 1] range
data = data.astype(np.float32)
data /= np.max(np.abs(data))  # Normalize

# Generate white noise signal
noise = np.random.normal(0, 1, len(data))

# Set adaptive filter parameters
mu = 0.01  # Step size
filter_order = 32  # Filter order
n_samples = len(data)

# Initialize filter weights and output
w = np.zeros(filter_order)
output = np.zeros(n_samples)
error = np.zeros(n_samples)

# Adaptive filter (LMS algorithm)
for i in range(filter_order, n_samples):
    x = noise[i-filter_order:i]
    d = data[i]
    y = np.dot(w, x)
    e = d - y
    w += 2 * mu * e * x
    output[i] = y
    error[i] = e

# Play the first 10 seconds of the original audio
print("Playing the first 10 seconds of the original audio")
playback = sa.play_buffer((data[:10*sample_rate] * 32767).astype(np.int16), 1, 2, sample_rate)
playback.wait_done()

# Play the first 2 seconds of the noisy audio
print("Playing the first 2 seconds of the noisy audio")
noisy_signal = data + noise
playback = sa.play_buffer((noisy_signal[:2*sample_rate] * 32767).astype(np.int16), 1, 2, sample_rate)
playback.wait_done()

# Play the first 10 seconds of the audio after adaptive noise cancellation
print("Playing the first 10 seconds of the audio after adaptive noise cancellation")
playback = sa.play_buffer((error[:10*sample_rate] * 32767).astype(np.int16), 1, 2, sample_rate)
playback.wait_done()

# Save the processed audio
wavfile.write(r'D:\AdaptiveNoiseCancelled.wav', sample_rate, (error * 32767).astype(np.int16))

# Visualize the original signal, noisy signal, and signal after adaptive noise cancellation
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(data)
plt.title('Original Signal')

plt.subplot(3, 1, 2)
plt.plot(noisy_signal)
plt.title('Noisy Signal')

plt.subplot(3, 1, 3)
plt.plot(error)
plt.title('Signal after Adaptive Noise Cancellation')

plt.tight_layout()
plt.show()

