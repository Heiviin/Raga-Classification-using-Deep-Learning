from flask import Flask, request, render_template_string
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment  # For handling mp3 to wav conversion
import json

app = Flask(__name__)

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


# HTML template for the front-end (embedded in the Python file)
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raga Classification</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            height: 100vh;  /* Ensure body takes up the full viewport height */
            background-image: url('static/image.jpg');  /* Replace with your image URL */
            background-size: cover;           /* Ensures the image covers the entire viewport */
            background-position: center;      /* Centers the image */
            background-repeat: no-repeat;     /* Prevents the image from repeating */
            color: white;
            text-align: center;
            display: flex;
            justify-content: flex-start;      /* Aligns content to the top */
            align-items: center;
            flex-direction: column;
            padding-top: 50px;  /* Adds space from the top */
        }

        h1 {
            font-size: 3em;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }

        form {
            margin-bottom: 20px;
        }

        input[type="file"] {
            font-size: 1.2em;
            padding: 10px;
            margin-bottom: 20px;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            border: none;
            border-radius: 5px;
        }

        button {
            font-size: 1.2em;
            padding: 10px 20px;
            background-color: rgba(255, 165, 0, 0.8);  /* Orange */
            border: none;
            border-radius: 5px;
            color: white;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: rgba(255, 140, 0, 0.8);
        }

        .predicted-raga {
            font-size: 2.5em;
            margin-top: 20px;
            font-weight: bold;
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.6);
            border-radius: 10px;
            animation: fadeIn 1s ease-out;
        }

        @keyframes fadeIn {
            0% {
                opacity: 0;
                transform: translateY(-20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .highlight {
            color: #ffcc00;  /* Highlight color */
            text-shadow: 2px 2px 6px rgba(255, 165, 0, 0.9);
        }
    </style>
</head>
<body>
    <h1>Raga Classification</h1>
    
    <!-- Form to upload the audio file -->
    <form method="POST" enctype="multipart/form-data">
        <label for="file">Choose an audio file (MP3/WAV):</label>
        <input type="file" name="file" required>
        <button type="submit">Predict</button>
    </form>

    <!-- Display the predicted raga -->
    {% if predicted_raga %}
        <div class="predicted-raga">
            Predicted Raga: <span class="highlight">{{ predicted_raga }}</span>
        </div>
    {% endif %}
</body>
</html>


"""


# Flask route for the home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded file
        file = request.files["file"]
        if file:
            # Save the file to a temporary location
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)

            # Predict the raga
            predicted_raga = predict_raga(file_path)

            # Return the result with the predicted raga
            return render_template_string(html_template, predicted_raga=predicted_raga)

    return render_template_string(html_template, predicted_raga=None)


if __name__ == "__main__":
    # Ensure the uploads directory exists
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    # Run the Flask app
    app.run(debug=True)
