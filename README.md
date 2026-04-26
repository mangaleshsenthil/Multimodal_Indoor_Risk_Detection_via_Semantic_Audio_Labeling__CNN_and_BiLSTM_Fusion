# Multimodal_Indoor_Risk_Detection_via_Semantic_Audio_Labeling__CNN_and_BiLSTM_Fusion

This repository contains the implementation of a multimodal deep learning framework for indoor risk detection. Traditional safety systems often fail because they rely on single-sensor thresholds (e.g., thermal alarms or sound detectors). This project addresses that by fusing environmental audio with temperature and humidity trends, detecting dangerous conditions that only become apparent when multiple sensory streams are analyzed together. 

## Key Features

- **Multimodal Fusion Engine:** Merges acoustic features and environmental sensor sequences (temperature, humidity) using a learned attention gate.
- **Semantic Risk Labeling:** Automatically assigns binary risk/danger labels to audio categories using Sentence-BERT transformer embeddings, eliminating the need for manual annotation.
- **Deep Learning Architectures:**
  - **Audio Branch:** A 3-layer Convolutional Neural Network (CNN) processes 128-band mel-frequency spectrograms.
  - **Sensor Branch:** A 2-layer Bidirectional Long Short-Term Memory (BiLSTM) network processes temporal gradient-augmented sensor readings.
- **Imbalance Handling:** Utilizes Focal Loss with Weighted Random Sampling to successfully train on real-world distributions where abnormal events are overwhelmingly outnumbered by normal events (9:1 ratio).

## Datasets Used

1. **[ESC-50 Dataset](https://github.com/karolpiczak/ESC-50):** 2,000 five-second audio recordings covering 50 environmental sound categories.
2. **[UCI Energy Appliances Dataset](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction):** Continuous indoor temperature and relative humidity recordings sampled at 10-minute intervals.

## Model Performance

The trained system achieves the following metrics on the test dataset:
- **Area Under ROC (AUC):** 0.916
- **Overall Accuracy:** 94%
- **Macro-Averaged F1 Score:** 0.79

## Repository Structure

- `multimodal_risk_detection.ipynb`: The main notebook containing the full data pipeline, semantic labeling, model definition, training loop, and evaluation metrics.
- `multimodal_risk_detection_paper.tex`: The LaTeX source code of the accompanying research paper outlining the architecture and experimental results.
- `best_model.pth`: Saved weights of the best-performing attention-based fusion model.
- `ESC-50/`: Directory containing the audio datasets (if downloaded).
- `energydata_complete.csv`: UCI Energy Appliances sensor logs.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- PyTorch
- Torchaudio
- Transformers (Hugging Face)
- Sentence-Transformers
- Librosa
- Scikit-learn
- Pandas, NumPy, Matplotlib

### Installation and Usage

1. Clone this repository to your local machine.
2. Download the required datasets (ESC-50 and UCI Energy). If you are using the included notebook, the data loading cells expect the data to be present in their respective directories.
3. Open `multimodal_risk_detection.ipynb` in Jupyter Notebook or Google Colab.
4. Run the cells sequentially to encode ESC-50 classes using Sentinel-BERT, extract mel-spectrograms, preprocess the sliding-window time-series data, and train the multimodal network.

## License

This project was developed as a partial fulfillment of coursework requirements (Semester 8, Department of Electronics and Communication Engineering). Please refer to the specific open-source licenses of the ESC-50 and UCI datasets for data usage policies.

