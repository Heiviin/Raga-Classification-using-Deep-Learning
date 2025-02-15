import librosa
import numpy as np
import os

# Define paths
audio_folder_path = r"C:\Users\SENSORS LAB-3\PycharmProjects\Ragam_classification_Rio\4_Raga_only_Vocals"
output_npz_path = "features_data.npz"

# Define parameters
sr = 16000  # Sample rate
n_mfcc = 19  # Number of MFCCs
n_chroma = 12  # Number of Chroma features
hop_length = int(0.75 * sr)  # 75% overlap
n_fft = 2048  # FFT size

# Lists to store features and labels
features_list = []
labels_list = []

# Preprocessing function for each audio file
def preprocess_audio(file_path):
    # Load audio file
    audio, _ = librosa.load(file_path, sr=sr)

    # 1. MFCC Features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # 2. Chroma Features (STFT)
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)

    # 3. Chroma CENS Features
    chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)

    # 4. Root Mean Square Energy (RMSE)
    rmse = librosa.feature.rms(y=audio)

    # 5. Spectral Features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)

    # 6. Zero-Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=audio)

    # 7. Pitch and Magnitude
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)

    # Combine all features into a single array
    combined_features = np.concatenate([
        mfccs,
        chroma_stft,
        chroma_cens,
        rmse,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zcr,
        pitches,
        magnitudes
    ], axis=0)

    return combined_features

# Process all files and collect features and labels
for subdir, _, files in os.walk(audio_folder_path):
    ragam_label = os.path.basename(subdir)
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(subdir, file)
            features = preprocess_audio(file_path)

            # Append features and labels to respective lists
            features_list.append(features)
            labels_list.append(ragam_label)

# Convert to NumPy arrays
features_array = np.array(features_list, dtype=object)  # Use object type for variable-length arrays
labels_array = np.array(labels_list)

# Save to .npz file
np.savez_compressed(output_npz_path, features=features_array, labels=labels_array)

print(f"Features and labels saved to {output_npz_path}.")
