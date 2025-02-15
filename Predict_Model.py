import librosa
import os
import numpy as np
from keras.models import load_model

num_mfcc = 13
n_fft = 2048
hop_length = 512
num_segments = 5
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

# Define the genre labels (ragas)
raga_directories =[
        "Harikamboji",
        "Kalyani",
        "Kharaharapriya",
        "Todi"
    ]

# Path to folder with test songs (e.g., 'kalyani' folder)
folder_path = r"C:\Users\SENSORS LAB-3\Downloads\Varnam  _ Vanajakshi - Ragam _ Kalyani ( Sing Along )"

# Load the pre-trained model
model = load_model('4_Raga_Vocal_model.keras')

# Variables to track correct predictions
total_files = 0
correct_predictions = 0

# Loop over each audio file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.wav'):  # Ensure you are reading only MP3 files
        total_files += 1
        file_path = os.path.join(folder_path, file_name)

        # Load the audio file
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

        # Compute MFCCs for each segment
        mfccs = []
        for s in range(num_segments):
            start_sample = s * samples_per_segment
            end_sample = (s + 1) * samples_per_segment
            mfcc = librosa.feature.mfcc(y=signal[start_sample:end_sample], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                        hop_length=hop_length)
            mfccs.append(mfcc.T)

        # Combine the MFCCs from all segments
        combined_mfccs = np.concatenate(mfccs, axis=0)

        # Ensure the combined MFCCs have exactly 130 time steps
        if combined_mfccs.shape[0] < 130:
            combined_mfccs = np.pad(combined_mfccs, ((0, 130 - combined_mfccs.shape[0]), (0, 0)), mode='constant')
        elif combined_mfccs.shape[0] > 130:
            combined_mfccs = combined_mfccs[:130, :]

        # Add batch dimension
        X = np.expand_dims(combined_mfccs, axis=0)

        # Make prediction
        prediction = model.predict(X)
        predicted_index = np.argmax(prediction, axis=1)

        # Extract the predicted raga
        predicted_raga = raga_directories[predicted_index[0]]

        # Check if the predicted raga matches the actual raga (assuming raga name is part of the filename)
        actual_raga = 'Kalyani'  # Since all songs are from Kalyani folder
        is_correct = predicted_raga == actual_raga

        # Print the result for each file
        print(f"File: {file_name} | Predicted: {predicted_raga} | Actual: {actual_raga} | Correct: {is_correct}")

        # Update correct predictions count
        if is_correct:
            correct_predictions += 1

# Calculate and print overall accuracy
if total_files > 0:
    accuracy = (correct_predictions / total_files) * 100
    print(f"Total_Files:{total_files}")
    print(f"Correct_predictions:{correct_predictions}")
    print(f"Overall Accuracy:{accuracy:.2f}%")
else:
    print("No audio files found in the folder.")



