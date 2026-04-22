# 🌿 PlantVillage Classification

A machine learning project for automated plant disease detection and classification using the PlantVillage dataset. This project employs multiple machine learning models including SVM, Random Forest, and deep learning (EfficientNet) to accurately identify diseases in tomato, potato, and pepper plants.

## 📋 Project Overview

This project classifies plant leaf images into 8 different disease categories:

**Tomato (4 classes):**
- Healthy
- Early Blight
- Late Blight
- Yellow Leaf Curl Virus

**Potato (2 classes):**
- Healthy
- Early Blight

**Pepper Bell (2 classes):**
- Healthy
- Bacterial Spot

## 🚀 Features

- **Multiple Classification Models:**
  - Support Vector Machine (SVM)
  - Random Forest Classifier
  - Deep Learning (EfficientNet CNN)

- **Image Preprocessing:**
  - Median filtering for noise reduction
  - Gaussian filtering options
  - Image resizing and normalization

- **Feature Extraction:**
  - GLCM (Gray-Level Co-occurrence Matrix) features
  - Hand-crafted texture features for ML models

- **Interactive Web Application:**
  - Streamlit-based web interface
  - Real-time disease prediction
  - Multiple model comparison
  - Image upload and analysis

## 📁 Project Structure

```
PlantVillage-classification/
├── notebook/                    # Jupyter notebook with analysis
│   └── PlantVillage.ipynb      # Data exploration and model training
├── streamlit_app/              # Web application
│   ├── app.py                  # Streamlit application
│   ├── efficientnet_final.h5   # Trained EfficientNet model
│   ├── svm_model.pkl           # Trained SVM model
│   ├── rf_model.pkl            # Trained Random Forest model
│   └── scaler.pkl              # Data scaler for preprocessing
├── features/                    # Preprocessed dataset
│   ├── X_train.npy             # Training features
│   ├── X_test.npy              # Test features
│   ├── y_train.npy             # Training labels
│   └── y_test.npy              # Test labels
├── figures/                     # Generated plots and visualizations
│   └── outputs/
├── test/                        # Test files and scripts
├── requirements.txt             # Python dependencies
└── rapport.pdf                  # Project report
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd PlantVillage-classification
   ```

2. **Create and activate virtual environment:**
   ```bash
   # On Windows
   python -m venv my_env
   my_env\Scripts\activate
   
   # On macOS/Linux
   python -m venv my_env
   source my_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Streamlit Web Application

```bash
streamlit run streamlit_app/app.py
```

The application will open in your browser at `http://localhost:8501`. You can:
- Upload plant leaf images
- Select different prediction models
- View classification results and confidence scores
- Compare predictions across multiple models

### Running the Jupyter Notebook

```bash
jupyter notebook notebook/PlantVillage.ipynb
```

The notebook contains:
- Data exploration and preprocessing
- Feature extraction and engineering
- Model training and evaluation
- Visualization of results

## 📦 Key Dependencies

- **TensorFlow/Keras**: Deep learning framework
- **OpenCV (cv2)**: Image processing
- **scikit-learn**: Machine learning algorithms
- **scikit-image**: Image analysis features (GLCM)
- **Streamlit**: Web application framework
- **NumPy/Pandas**: Data manipulation
- **Matplotlib**: Data visualization

See `requirements.txt` for complete dependency list.

## 🤖 Models

### 1. Support Vector Machine (SVM)
- Uses hand-crafted GLCM texture features
- Effective for small to medium datasets
- Fast inference time

### 2. Random Forest
- Ensemble learning approach
- Robust to overfitting
- Feature importance analysis available

### 3. EfficientNet (Deep Learning)
- Pre-trained CNN architecture
- Transfer learning approach
- Higher accuracy on complex patterns
- Best overall performance

## 📊 Model Performance

Models are evaluated on test set using:
- Accuracy
- Precision, Recall, F1-Score
- Confusion matrices
- ROC curves

## 🔍 Image Preprocessing

Images undergo the following preprocessing steps:
1. **Filtering**: Median blur to remove noise
2. **Resizing**: Standardized to 224×224 pixels
3. **Normalization**: Feature scaling using scaler.pkl
4. **Segmentation**: Optional background removal

## 🐛 Troubleshooting

- **Model not found error**: Ensure all `.h5` and `.pkl` files are present in `streamlit_app/`
- **Memory issues**: Reduce batch size or process images sequentially
- **Slow inference**: Use GPU acceleration if available (CUDA)

## 📚 Dataset

The project uses the **PlantVillage Dataset**, which contains over 50,000 high-quality images of healthy and diseased plant leaves.

**Citation**: Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv preprint arXiv:1504.04082.

## 📝 License

This project is provided as-is for educational and research purposes.

## 👤 Author

Created as an agricultural technology project for automated plant disease detection.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Improve model accuracy

## 📧 Support

For questions or issues, please refer to the project documentation or submit an issue in the repository.

---

**Note**: This project demonstrates the application of machine learning and deep learning techniques to real-world agricultural problems, providing a practical tool for farmers and agricultural scientists to quickly identify and manage plant diseases.
