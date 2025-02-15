import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json

# Load features and labels from the JSON file
with open("features.json", "r") as json_file:
    data = json.load(json_file)

# Prepare the dataset
X = []
y = []

# Extract features and labels
for feature in data["features"]:
    X.append(feature["mfcc"] + feature["chroma_stft"] + feature["chroma_cens"] +
             [feature["rmse"], feature["spectral_centroid"], feature["spectral_bandwidth"],
              feature["spectral_rolloff"], feature["zcr"], feature["pitch_mean"], feature["magnitude_mean"]])
    y.append(feature["label"])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels to integers (assuming labels are strings)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Reshape X to match LSTM input requirements (samples, time_steps, features)
# If the data does not have an actual time-series structure, we can add an extra dimension as time_steps=1
X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape to (samples, time_steps=1, features)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the simplified LSTM model
model = models.Sequential([
    layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'),  # Single LSTM layer
    layers.Dense(4, activation='softmax')  # Output layer for 4 raga classes
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    epochs=50,  # Reduced epochs for faster training
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    verbose=1)

# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=1)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Save the model
model.save("ragam_classification_lstm_model.keras")
