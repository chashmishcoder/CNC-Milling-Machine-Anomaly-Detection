# 🤖 CNC Milling Machine Anomaly Detection

## 🌟 Overview

This project implements an advanced anomaly detection system for CNC milling machines using cutting-edge deep learning techniques. The application leverages an autoencoder neural network to analyze vibration data and identify potential machine performance issues in real-time.

## ✨ Key Features

- 🕵️ **Anomaly Detection**: Uses an autoencoder model to detect machine irregularities
- 📊 **Tri-axial Vibration Analysis**: Processes 4096-point vibration data across three axes
- 🖥️ **User-Friendly Interface**: Streamlit-based web application for easy interaction
- 📈 **Performance Metrics**: Detailed model performance tracking
- 🎯 **Dynamic Threshold Calculation**: Adaptive anomaly detection mechanism

## 📚 Dataset Citation

**Original Dataset Source**: 
Tnani, M., Feil, M., & Diepold, K. (2022). Smart Data Collection System for Brownfield CNC milling Machines: A new benchmark dataset for Data-Driven Machine Monitoring. Procedia CIRP, 107, 131–136. https://doi.org/10.1016/j.procir.2022.04.022

**Original Repository**: [[Link to GitHub Repository of Original Dataset](https://github.com/boschresearch/CNC_Machining)]

If you use this dataset or model in your research, please cite the original work

## 📊 Model Performance Metrics

- **False Positive Rate**: 1.2%
- **Recall (Anomalies)**: 86%
- **Precision**: 84%
- **F1 Score**: 85%
- **Specificity**: 98%

## 🚀 Live Deployment

👉 **Access the Application**: [CNC Anomaly Detector Live Demo](https://cncanomalyproject.streamlit.app/)

## 💻 Prerequisites

- Python 3.8+
- Required Libraries:
  - Streamlit
  - TensorFlow
  - NumPy
  - Pandas
  - Plotly
  - scikit-learn
  - h5py
  - joblib

## 🛠️ Installation

1. Clone the repository
```bash
git clone https://github.com/YourUsername/CNC-Anomaly-Detection.git
cd CNC-Anomaly-Detection
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## 🖱️ Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

## 📤 File Upload Requirements

- Supports .h5 and .csv file formats
- Expects tri-axial vibration data
- Recommended data length: 4096 points

## 🏭 Model Training Details

- **Training Duration**: 12 Hours
- **Machines Used**: 3
- **Training Data Span**: 2 Years
- **Validation Set Size**: 408 Samples

## 🔍 Anomaly Detection Mechanism

The system uses an autoencoder to reconstruct input vibration signals. Reconstruction errors are compared against a dynamically calculated threshold (95th percentile of training data errors) to identify potential anomalies.

## 🤝 Contributing

Contributions are welcome! Please submit pull requests or open issues to suggest improvements or report bugs.

## 📬 Contact

Developed by [Your Name/GitHub Username]
- 🌐 GitHub: [Your GitHub Profile Link]
- 📧 Email: [Your Contact Email]

## 🙏 Acknowledgments

- Gratitude to the original researchers for providing the CNC vibration dataset
- Inspiration from industrial machine learning applications
- Open-source libraries that made this project possible

---

