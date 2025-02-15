import librosa
import numpy as np
import pickle
import os

# Define paths
audio_folder_path = r"C:\Users\SENSORS LAB-3\PycharmProjects\Ragam_classification_Rio\4_Raga_only_Vocals"
output_pickle_path = "features.pkl"

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

    # Concatenate all features into a dictionary
    features = {
        "mfcc": mfcc_mean.tolist(),  # 19 MFCC features
        "chroma_stft": chroma_stft_mean.tolist(),  # 12 Chroma STFT features
        "chroma_cens": chroma_cens_mean.tolist(),  # 12 Chroma CENS features
        "rmse": rmse_mean,  # 1 RMSE
        "spectral_centroid": spectral_centroid_mean,  # Mean Spectral Centroid
        "spectral_bandwidth": spectral_bandwidth_mean,  # Mean Spectral Bandwidth
        "spectral_rolloff": spectral_rolloff_mean,  # Mean Spectral Rolloff
        "zcr": zcr_mean,  # Mean ZCR
        "pitch_mean": pitch_mean,  # Mean pitch
        "magnitude_mean": magnitude_mean,  # Mean magnitude
        "label": ragam_label  # Label for the raga
    }

    return features

# Process all files and add to data dictionary
for subdir, _, files in os.walk(audio_folder_path):
    ragam_label = os.path.basename(subdir)
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(subdir, file)
            features = preprocess_audio(file_path, ragam_label)

            # Append features to the data dictionary
            data["features"].append(features)

# Save features to a single pickle file
with open(output_pickle_path, "wb") as pickle_file:
    pickle.dump(data, pickle_file)

print("Data preprocessing with 50 features saved to a single pickle file completed.")
