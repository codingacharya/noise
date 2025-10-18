import numpy as np
import soundfile as sf

sr = 16000
duration = 3  # 3 seconds
t = np.linspace(0, duration, int(sr*duration), endpoint=False)

# Simulated "clean" tone (speech-like)
clean = 0.5 * np.sin(2 * np.pi * 220 * t)

# Add white Gaussian noise to create a "noisy" version
noise = 0.05 * np.random.randn(len(t))
noisy = clean + noise

# Save both
sf.write("clean_sample.wav", clean, sr)
sf.write("noisy_sample.wav", noisy, sr)

print("✅ Generated clean_sample.wav and noisy_sample.wav")
