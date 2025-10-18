import streamlit as st
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import butter, lfilter
import noisereduce as nr
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility

st.set_page_config(page_title="AI Noise Cancellation Demo", layout="wide")
st.title("🎧 AI-Powered Noise Cancellation Demo (Noisereduce)")

# ----------------------------
# DSP Utilities
# ----------------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def compute_metrics(clean, processed, sr):
    clean_t = torch.tensor(clean).unsqueeze(0)
    proc_t = torch.tensor(processed).unsqueeze(0)
    stoi_val = short_time_objective_intelligibility(proc_t, clean_t, sr).item()

    # Proxy "quality" metric based on spectral difference
    clean_spec = np.abs(librosa.stft(clean))
    proc_spec = np.abs(librosa.stft(processed))
    spectral_diff = np.mean(np.abs(clean_spec - proc_spec))
    quality_score = max(0, 5 - spectral_diff * 100)  # pseudo 0–5 scale

    snr_val = 10 * np.log10(np.sum(clean**2) / np.sum((clean - processed)**2 + 1e-8))
    return quality_score, stoi_val, snr_val

# ----------------------------
# Sidebar: DSP Parameters
# ----------------------------
st.sidebar.header("DSP Parameters")
lowcut = st.sidebar.slider("Low Cutoff Frequency (Hz)", 20, 500, 80)
highcut = st.sidebar.slider("High Cutoff Frequency (Hz)", 3000, 16000, 7900)
filter_order = st.sidebar.slider("Filter Order", 1, 10, 5)

# ----------------------------
# Audio Input
# ----------------------------
st.header("1️⃣ Upload or Record Audio")
col1, col2 = st.columns(2)
with col1:
    clean_file = st.file_uploader("Upload Clean Speech", type=["wav", "mp3"])
with col2:
    noisy_file = st.file_uploader("Upload Noisy Speech", type=["wav", "mp3"])

record = st.checkbox("🎤 Record Live Audio (Noisy)")
if record:
    duration = st.slider("Recording Duration (seconds)", 1, 10, 3)
    sr = 16000
    st.info(f"Recording {duration}s of audio...")
    recording = sd.rec(int(duration*sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    noisy = recording.flatten()
    sf.write("recorded_noisy.wav", noisy, sr)
    st.audio("recorded_noisy.wav", format="audio/wav")
else:
    noisy = None
    sr = 16000

# ----------------------------
# Run Denoising
# ----------------------------
ai_option = st.checkbox("✅ Use AI-Based Denoising (Noisereduce)", value=True)

if st.button("🚀 Apply Noise Cancellation"):
    # Load audio
    if clean_file:
        clean, sr_c = librosa.load(clean_file, sr=None)
    elif noisy is not None:
        clean = noisy.copy()
        sr_c = sr
    else:
        st.warning("Upload or record audio first.")
        st.stop()

    if noisy_file:
        noisy, sr_n = librosa.load(noisy_file, sr=None)
        sr = min(sr_c, sr_n)
        clean = librosa.resample(y=clean, orig_sr=sr_c, target_sr=sr)
        noisy = librosa.resample(y=noisy, orig_sr=sr_n, target_sr=sr)
    elif record:
        sr = 16000
    else:
        st.warning("No noisy audio provided.")
        st.stop()

    # Trim to same length
    length = min(len(clean), len(noisy))
    clean, noisy = clean[:length], noisy[:length]

    # Apply selected denoising
    if ai_option:
        st.info("Applying AI-based denoising (Noisereduce)...")
        denoised = nr.reduce_noise(y=noisy.astype(np.float32), sr=sr)
    else:
        st.info("Applying baseline DSP bandpass filter...")
        denoised = bandpass_filter(noisy, lowcut, highcut, sr, filter_order)

    # Save denoised
    sf.write("denoised.wav", denoised, sr)

    # Metrics
    quality_val, stoi_val, snr_val = compute_metrics(clean, denoised, sr)

    # ----------------------------
    # Display Metrics
    # ----------------------------
    st.subheader("📊 Quality Metrics After Denoising")
    c1, c2, c3 = st.columns(3)
    c1.metric("Quality (proxy)", f"{quality_val:.3f} / 5")
    c2.metric("STOI", f"{stoi_val:.3f}")
    c3.metric("SNR (dB)", f"{snr_val:.2f}")

    # ----------------------------
    # Audio Playback
    # ----------------------------
    st.subheader("🎧 Listen to Audio")
    st.audio("denoised.wav", format="audio/wav")
    if noisy_file or record:
        st.audio("recorded_noisy.wav" if record else noisy_file, format="audio/wav")
    if clean_file:
        st.audio(clean_file, format="audio/wav")

    # ----------------------------
    # Waveform & Spectrogram
    # ----------------------------
    st.subheader("📈 Waveforms & Spectrograms")
    fig, axes = plt.subplots(3,2, figsize=(12,8))
    for i, data, title in zip(range(3), [clean, noisy, denoised], ["Clean", "Noisy", "Denoised"]):
        axes[i,0].plot(data)
        axes[i,0].set_title(f"{title} Waveform")
        axes[i,1].specgram(data, Fs=sr)
        axes[i,1].set_title(f"{title} Spectrogram")
    plt.tight_layout()
    st.pyplot(fig)

    # ----------------------------
    # Download
    # ----------------------------
    with open("denoised.wav","rb") as f:
        st.download_button("⬇️ Download Denoised Audio", f, file_name="denoised.wav")
