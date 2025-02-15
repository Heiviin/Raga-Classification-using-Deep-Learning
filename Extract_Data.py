import os
import librosa
import pandas as pd
import json
import numpy as np

# Path to the dataset containing audio files (segmented)
DATASET_PATH = "5_Raga_dataset_250_segments"  # Update with the correct path to your dataset
JSON_PATH = "data_audio.json"

# Columns for the dataset
columns = ['filename', 'rmse', 'chroma_stft', 'spec_cent', 'spec_bw', 'rolloff', 'zcr',
           'mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7',
           'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'raga']

# Initialize an empty DataFrame
dataset = pd.DataFrame(columns=columns)


def save_audio_features(dataset_path, json_path):
    """Extract features from audio files and save them to a JSON file."""
    global dataset  # Declare dataset as a global variable

    # Dictionary to store the data
    data = {
        "mapping": [],
        "labels": [],
        "audio_features": []
    }

    # Get the list of ragas (subdirectories)
    ragas = os.listdir(dataset_path)

    for raga in ragas:
        path = os.path.join(dataset_path, raga)
        musics = os.listdir(path)

        # Loop through all audio files in the raga folder
        for name in musics:
            vocals = os.path.join(path, name)  # Correct path to the segmented audio file
            filename = name

            # Load the segmented audio file
            try:
                y, sr = librosa.load(vocals, mono=True)

                # Extract audio features
                rmse = librosa.feature.rms(y=y)[0]
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)

                # Data to append
                data_row = [filename, np.mean(rmse), np.mean(chroma_stft), np.mean(spec_cent), np.mean(spec_bw),
                            np.mean(rolloff), np.mean(zcr), np.mean(mfcc[0]), np.mean(mfcc[1]), np.mean(mfcc[2]),
                            np.mean(mfcc[3]), np.mean(mfcc[4]), np.mean(mfcc[5]), np.mean(mfcc[6]), np.mean(mfcc[7]),
                            np.mean(mfcc[8]), np.mean(mfcc[9]), np.mean(mfcc[10]), np.mean(mfcc[11]), np.mean(mfcc[12]),
                            np.mean(mfcc[13]), raga]

                # Append data to the DataFrame
                dataset = pd.concat([dataset, pd.DataFrame([data_row], columns=dataset.columns)], ignore_index=True)

                # Store features in the dictionary for JSON output
                data["audio_features"].append({
                    "filename": filename,
                    "features": {
                        "rmse": np.mean(rmse).tolist(),
                        "chroma_stft": np.mean(chroma_stft).tolist(),
                        "spec_cent": np.mean(spec_cent).tolist(),
                        "spec_bw": np.mean(spec_bw).tolist(),
                        "rolloff": np.mean(rolloff).tolist(),
                        "zcr": np.mean(zcr).tolist(),
                        "mfcc": [np.mean(mfcc[i]).tolist() for i in range(mfcc.shape[0])]
                    },
                    "raga": raga
                })

                data["mapping"].append(raga)
                data["labels"].append(raga)

                print(f"{filename} ({raga}) added successfully.")

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")

    # Save features to a JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_audio_features(DATASET_PATH, JSON_PATH)
