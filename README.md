 🎵 Raga Classification Using Deep Learning 🎶

## 🌟 Overview
This project explores deep learning techniques for classifying Indian classical ragas. Unlike Western genre classification, raga classification presents unique challenges due to its modal nature, improvisational aspects, and limited dataset availability. This study aims to overcome these challenges by employing specialized feature extraction and deep learning models.

## 🔥 Features
- **📂 Dataset**: Curated raga dataset with vocal-only and vocal + instrumental segments.
- **🧠 Feature Extraction**: 
  -  MFCC
  -  Chroma CENS
  -  Chroma STFT
  -  RMSE
  -  Spectral Centroid
  - Spectral Bandwidth
  -  Spectral Rolloff
  -  Zero Crossing Rate (ZCR)
  -  Pitch Mean
  -  Magnitude Mean
- **🤖 Deep Learning Models**:
  -  LSTM Model
  -  ANN Model with NPZ Data Storage
  -  BERT-Based Transformer Model
  -  Final ANN Model without Feature Averaging
- **🛠 Tools Used**:
  - 🎵 `librosa` for feature extraction
  - 🎙 `Demucs` for vocal isolation
  - 🤖 `TensorFlow/Keras` for deep learning models
  - 📊 `NumPy` and `Pandas` for data handling
  - 📁 `JSON` and `NPZ` for efficient data storage

## 📀 Dataset Preparation
1. **📥 Data Collection**: Raga audio recordings were segmented into meaningful phrases.
2. **🎙 Vocal Isolation**: Used Demucs to separate vocal components.
3. **📊 Feature Extraction**: Stored extracted features in JSON and NPZ formats.

## 📈 Model Performance
| 🏗 Model | 🎯 Accuracy |
|--------|----------|
| LSTM | 34.80% |
| ANN with NPZ | 30% |
| BERT-Based Transformer | 33.1% |
| Final ANN (Vocal + Instrument) | 53% |
| Final ANN (Vocal-Only) | 73% |

## 🧐 Results and Analysis
- ❌ Traditional Western classification methods yielded poor accuracy (e.g., 22% using only MFCCs).
- ✅ Expanded feature extraction improved model performance.
- 🎤 Vocal-only datasets led to the highest accuracy (73%).
- 🏆 ANN models performed better than LSTMs and Transformers.

## ⚙️ Installation & Usage
### 📌 Prerequisites
- 🐍 Python 3.8+
- 🤖 TensorFlow/Keras
- 🎵 Librosa
- 📊 NumPy, Pandas
- 🎙 Demucs (for vocal isolation)


## 🔮 Future Scope
- 📚 Expanding the dataset with more ragas and labeled samples.
- 🏗️ Exploring hybrid deep learning models.
- 🎼 Incorporating musicological insights to refine classification techniques.

## 🤝 Contributors
- **Your Name** ([karthickrio1002@gmail.com])


