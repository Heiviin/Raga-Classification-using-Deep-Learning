import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json

# Load features and labels from the JSON file
with open("3_raga_features.json", "r") as json_file:
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

# Encode labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalize features
X_train = X_train / np.max(np.abs(X_train), axis=0)
X_val = X_val / np.max(np.abs(X_train), axis=0)  # Use training normalization for validation

# Define a simpler ANN model
from tensorflow.keras import layers, models

# Define the model
model = models.Sequential()

# Input layer and first hidden layer
model.add(layers.Dense(512, activation='relu', input_dim=X.shape[1],))  # Replace 'input_dim' with the number of input features
model.add(layers.Dropout(0.5))  # Dropout layer for regularization

# Second hidden layer
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting

# Third hidden layer
model.add(layers.Dense(128, activation='relu'))

# Output layer
model.add(layers.Dense(4, activation='softmax'))  # Change the number '4' based on the number of ragas

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summarize the model architecture
model.summary()


# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=1)
print(f'Validation accuracy: {val_accuracy * 100:.2f}%')

# Save the trained model
model.save("3_ragam_classification_ann_model.keras")
