import librosa
import numpy as np
import json
import os
from pydub import AudioSegment

# Define paths
audio_folder_path = r"C:\Users\SENSORS LAB-3\PycharmProjects\Ragam_classification_Rio\3_Raga_dataset_only_vocals"
output_json_path = "3_raga_features.json"

# Define parameters
sr = 16000  # Sample rate
n_mfcc = 19  # Number of MFCCs
n_chroma = 12  # Number of Chroma features
hop_length = int(0.75 * sr)  # 75% overlap
n_fft = 2048  # FFT size

# Data dictionary to store all features
data = {
    "features": []
}

# Helper function to convert mp3 to wav
def convert_mp3_to_wav(mp3_path):
    sound = AudioSegment.from_mp3(mp3_path)
    wav_path = mp3_path.replace(".mp3", ".wav")
    sound.export(wav_path, format="wav")
    return wav_path

# Preprocessing function for each audio file
def preprocess_audio(file_path, ragam_label):
    # Load audio file
    audio, _ = librosa.load(file_path, sr=sr)

    # 1. MFCC Features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)  # Mean of each MFCC

    # 2. Chroma Features (STFT)
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft, axis=1)  # Mean of each chroma feature

    # 3. Chroma CENS Features
    chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
    chroma_cens_mean = np.mean(chroma_cens, axis=1)  # Mean of each chroma cens feature

    # 4. Root Mean Square Energy (RMSE)
    rmse = librosa.feature.rms(y=audio)
    rmse_mean = np.mean(rmse)  # Mean RMSE value

    # 5. Spectral Features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)

    spectral_centroid_mean = np.mean(spectral_centroid)  # Mean Spectral Centroid
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)  # Mean Spectral Bandwidth
    spectral_rolloff_mean = np.mean(spectral_rolloff)  # Mean Spectral Rolloff

    # 6. Zero-Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_mean = np.mean(zcr)  # Mean ZCR

    # 7. Pitch and Magnitude Mean
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0  # Mean pitch where pitch exists
    magnitude_mean = np.mean(magnitudes[magnitudes > 0]) if np.any(magnitudes > 0) else 0  # Mean magnitude

    # Convert numpy.float32 values to standard Python floats
    features = {
        "mfcc": mfcc_mean.tolist(),  # 19 MFCC features
        "chroma_stft": chroma_stft_mean.tolist(),  # 12 Chroma STFT features
        "chroma_cens": chroma_cens_mean.tolist(),  # 12 Chroma CENS features
        "rmse": float(rmse_mean),  # 1 RMSE
        "spectral_centroid": float(spectral_centroid_mean),  # Mean Spectral Centroid
        "spectral_bandwidth": float(spectral_bandwidth_mean),  # Mean Spectral Bandwidth
        "spectral_rolloff": float(spectral_rolloff_mean),  # Mean Spectral Rolloff
        "zcr": float(zcr_mean),  # Mean ZCR
        "pitch_mean": float(pitch_mean),  # Mean pitch
        "magnitude_mean": float(magnitude_mean),  # Mean magnitude
        "label": ragam_label  # Label for the raga
    }

    return features

# Process all files and add to data dictionary
for subdir, _, files in os.walk(audio_folder_path):
    ragam_label = os.path.basename(subdir)
    for file in files:
        if file.endswith(".wav") or file.endswith(".mp3"):
            file_path = os.path.join(subdir, file)
            if file.endswith(".mp3"):
                file_path = convert_mp3_to_wav(file_path)  # Convert mp3 to wav

            features = preprocess_audio(file_path, ragam_label)

            # Append features to the data dictionary
            data["features"].append(features)

# Save features to JSON file
with open(output_json_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

print("Data preprocessing with features saved to JSON completed.")
