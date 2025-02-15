import os
import math
import librosa
import json
import numpy as np

DATASET_PATH = "4_Raga_dataset_250_segments"
JSON_PATH = "5_features_.json"




def save_features(dataset_path, json_path, num_segments=5):
    data = {
        "mapping": [],
        "labels": [],
        "features": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    hop_length = 512
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                for d in range(num_segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    segment_features = extract_features(signal[start:finish], sample_rate, hop_length)

                    if segment_features:  # Only add if features are successfully extracted
                        data["features"].append(segment_features)
                        data["labels"].append(i - 1)

    # Convert numpy types to Python types before saving to JSON
    data = convert_to_python_type(data)

    # Save features and labels to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def extract_features(segment, sr, hop_length):
    features = {}

    # 1. MFCC features
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, hop_length=hop_length)
    features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()

    # 2. Pitch features
    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    features['pitch_mean'] = np.mean(pitches)
    features['pitch_std'] = np.std(pitches)

    # 3. Chroma (tonal) features
    chroma = librosa.feature.chroma_cens(y=segment, sr=sr)
    features['chroma_mean'] = np.mean(chroma, axis=1).tolist()

    # 4. Tempo (rhythm) features
    onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    features['tempo'] = tempo

    # 5. Spectral rolloff (frequency-related) features
    spec_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    features['rolloff_mean'] = np.mean(spec_rolloff)

    return features


def convert_to_python_type(obj):
    """Recursively converts numpy types to Python native types"""
    if isinstance(obj, dict):
        return {key: convert_to_python_type(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_type(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)  # Convert numpy floats to Python float
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)  # Convert numpy ints to Python int
    else:
        return obj


if __name__ == "__main__":
    save_features(DATASET_PATH, JSON_PATH, num_segments=10)
