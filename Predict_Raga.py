import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment  # For handling mp3 to wav conversion
import json

# Load the trained model
model_path = "ragam_classification_ann_model1.keras"
model = tf.keras.models.load_model(model_path)

# Load the label encoder used for training
with open("features.json", "r") as json_file:
    data = json.load(json_file)

labels = [feature["label"] for feature in data["features"]]
label_encoder = LabelEncoder()
label_encoder.fit(labels)  # Fit the label encoder with the labels used in training

# Preprocess a single audio file for prediction
def preprocess_for_prediction(file_path, sr=16000, n_mfcc=19, n_chroma=12, n_fft=2048, hop_length=12000):
    # Load audio
    audio, _ = librosa.load(file_path, sr=sr)

    # Extract features (same as in the training code)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
    rmse = librosa.feature.rms(y=audio)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)

    # Compute mean values
    mfcc_mean = np.mean(mfccs, axis=1)
    chroma_stft_mean = np.mean(chroma_stft, axis=1)
    chroma_cens_mean = np.mean(chroma_cens, axis=1)
    rmse_mean = np.mean(rmse)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_rolloff_mean = np.mean(spectral_rolloff)
    zcr_mean = np.mean(zcr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    magnitude_mean = np.mean(magnitudes[magnitudes > 0]) if np.any(magnitudes > 0) else 0

    # Combine all features into a single array
    features = mfcc_mean.tolist() + chroma_stft_mean.tolist() + chroma_cens_mean.tolist() + [
        rmse_mean, spectral_centroid_mean, spectral_bandwidth_mean, spectral_rolloff_mean, zcr_mean,
        pitch_mean, magnitude_mean
    ]

    # Reshape features for model input (1 sample, features)
    features = np.array(features).reshape(1, -1)  # Correct to 2D array (1, 50)
    return features

# Function to convert mp3 to wav
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

# Predict raga for a single audio file
def predict_raga(file_path):
    # If it's an mp3, convert it to wav
    if file_path.endswith(".mp3"):
        wav_file_path = file_path.replace(".mp3", ".wav")
        convert_mp3_to_wav(file_path, wav_file_path)
        file_path = wav_file_path  # Use the converted wav file

    # Preprocess the audio file
    features = preprocess_for_prediction(file_path)

    # Make prediction
    predictions = model.predict(features)
    predicted_label = np.argmax(predictions)
    predicted_ragam = label_encoder.inverse_transform([predicted_label])[0]

    return predicted_ragam

# Example usage
file_path = r"C:\Users\SENSORS LAB-3\PycharmProjects\Ragam_classification_Rio\4_Raga_only_Vocals\Harikamboji\Hari_CC_Alapana_segment_1_vocals.wav"
predicted_raga = predict_raga(file_path)
print(f"Predicted Raga: {predicted_raga}")

