import librosa
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set parameters
sr = 16000  # Sample rate (can be adjusted based on your dataset)
n_mfcc = 13  # Number of MFCC coefficients
max_length = 500  # Maximum sequence length for padding

def preprocess_audio(file_path, ragam_label):
    """
    Preprocess audio files to extract raw frame-level features without averaging.
    """
    audio, _ = librosa.load(file_path, sr=sr)

    # Extract raw frame features without averaging
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
    rmse = librosa.feature.rms(y=audio)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)

    # Store the frames directly as lists (no averaging)
    features = {
        "mfcc": mfccs.T.tolist(),  # Transpose to have time axis first
        "chroma_stft": chroma_stft.T.tolist(),
        "chroma_cens": chroma_cens.T.tolist(),
        "rmse": rmse.T.tolist(),
        "spectral_centroid": spectral_centroid.T.tolist(),
        "spectral_bandwidth": spectral_bandwidth.T.tolist(),
        "spectral_rolloff": spectral_rolloff.T.tolist(),
        "zcr": zcr.T.tolist(),
        "pitch": pitches.T.tolist(),
        "magnitude": magnitudes.T.tolist(),
        "label": ragam_label
    }

    return features


def preprocess_dataset(file_paths, labels):
    """
    Preprocess multiple audio files and return padded feature arrays.
    """
    X_padded = []
    y_labels = []

    for file_path, label in zip(file_paths, labels):
        feature = preprocess_audio(file_path, label)

        # Padding each feature to max_length
        mfcc_padded = pad_sequences([feature["mfcc"]], maxlen=max_length, padding='post', truncating='post')
        chroma_stft_padded = pad_sequences([feature["chroma_stft"]], maxlen=max_length, padding='post', truncating='post')
        chroma_cens_padded = pad_sequences([feature["chroma_cens"]], maxlen=max_length, padding='post', truncating='post')
        rmse_padded = pad_sequences([feature["rmse"]], maxlen=max_length, padding='post', truncating='post')
        spectral_centroid_padded = pad_sequences([feature["spectral_centroid"]], maxlen=max_length, padding='post', truncating='post')
        spectral_bandwidth_padded = pad_sequences([feature["spectral_bandwidth"]], maxlen=max_length, padding='post', truncating='post')
        spectral_rolloff_padded = pad_sequences([feature["spectral_rolloff"]], maxlen=max_length, padding='post', truncating='post')
        zcr_padded = pad_sequences([feature["zcr"]], maxlen=max_length, padding='post', truncating='post')

        # Concatenate all features into a single array
        padded_features = np.concatenate([
            mfcc_padded,
            chroma_stft_padded,
            chroma_cens_padded,
            rmse_padded,
            spectral_centroid_padded,
            spectral_bandwidth_padded,
            spectral_rolloff_padded,
            zcr_padded
        ], axis=-1)

        # Append the padded features and the label
        X_padded.append(padded_features)
        y_labels.append(feature["label"])

    # Convert to numpy arrays
    X = np.array(X_padded)
    y = np.array(y_labels)

    return X, y

from tensorflow.keras import layers, models

def build_ann_model(input_shape, num_classes):
    """
    Build a simple Artificial Neural Network (ANN) for sequence classification.
    """
    model = models.Sequential()

    # Flatten input data
    model.add(layers.Flatten(input_shape=input_shape))

    # Add hidden layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout layer for regularization
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout layer for regularization

    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))  # num_classes is the number of ragas

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.save("ragam_classification_feature_full_ann_model1.keras")

# Assuming you have your file paths and corresponding labels
file_paths = r"C:\Users\SENSORS LAB-3\PycharmProjects\Ragam_classification_Rio\4_Raga_only_Vocals"
labels = [0, 1, 2, ...]  # Corresponding labels for each file (e.g., raga IDs)

# Preprocess the data
X, y = preprocess_dataset(file_paths, labels)

# Define input shape and number of classes
input_shape = X.shape[1:]  # Shape of one input sample (sequence length, num features)
num_classes = len(np.unique(y))  # Number of ragas

# Build and train the ANN model
ann_model = build_ann_model(input_shape, num_classes)
ann_model.summary()
history = ann_model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = ann_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

