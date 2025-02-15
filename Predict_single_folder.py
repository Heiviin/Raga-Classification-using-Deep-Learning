import librosa
import numpy as np
from keras.models import load_model
import os

# Configuration for audio processing
sr = 16000  # Sample rate
n_mfcc = 19  # Number of MFCCs
n_chroma = 12  # Number of Chroma features
hop_length = int(0.75 * sr)  # 75% overlap
n_fft = 2048  # FFT size

# Define the raga labels (these should match your model's output order)
raga_labels = ["Harikamboji", "Kalyani", "Kharaharapriya", "Todi"]

# Load the pre-trained model
model = load_model('ragam_classification_ann_model1.keras')

# Define the target raga
target_raga = "Kharaharapriya"  # Set the target raga you want to evaluate

# Feature extraction function
def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=sr)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)

    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft, axis=1)

    chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
    chroma_cens_mean = np.mean(chroma_cens, axis=1)

    rmse = librosa.feature.rms(y=audio)
    rmse_mean = np.mean(rmse)

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)

    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_rolloff_mean = np.mean(spectral_rolloff)

    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_mean = np.mean(zcr)

    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0
    magnitude_mean = np.mean(magnitudes[magnitudes > 0]) if np.any(magnitudes > 0) else 0.0

    features = np.concatenate([
        mfcc_mean, chroma_stft_mean, chroma_cens_mean,
        [rmse_mean, spectral_centroid_mean, spectral_bandwidth_mean, spectral_rolloff_mean, zcr_mean, pitch_mean, magnitude_mean]
    ])

    return features.reshape(1, -1)

# Prediction and evaluation function for the target raga
def evaluate_target_raga(folder_path, target_raga):
    correct_predictions = 0
    total_files = 0

    # Process each audio file in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            features = extract_features(file_path)

            # Predict using the model
            prediction = model.predict(features)
            predicted_index = np.argmax(prediction, axis=1)
            predicted_raga = raga_labels[predicted_index[0]]

            # Compare prediction with the target raga
            if predicted_raga == target_raga:
                correct_predictions += 1
            total_files += 1

            # Print individual prediction result
            print(f"File: {file}, Target Raga: {target_raga}, Predicted Raga: {predicted_raga}")

    # Calculate accuracy for the target raga
    accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0
    print(f"\nTotal files: {total_files}")
    print(f"Correct predictions for '{target_raga}': {correct_predictions}")
    print(f"Accuracy for '{target_raga}': {accuracy:.2f}%")

# Example usage
folder_path =r"C:\Users\SENSORS LAB-3\Downloads\Kharaharapriya_raga_test\Segments"
evaluate_target_raga(folder_path, target_raga)
