
# 🧠 Brain Tumor Classification Using Deep Learning Algorithms

This project aims to classify brain tumors using deep learning models such as Convolutional Neural Networks (CNNs). Leveraging medical image datasets, the notebook trains and evaluates models to detect different types of brain tumors, helping advance the field of automated diagnostics in medical imaging.

## 📂 Project Structure

```
Brain-Tumor-Classification/
│
├── Brain Tumor Classification Using Deep Learning Algorithms.ipynb
├── /dataset
│   ├── glioma
│   ├── meningioma
│   ├── no_tumor
│   └── pituitary
├── /models
│   └── saved_model.h5
├── /outputs
│   └── visualizations, reports, metrics
└── README.md
```

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- OpenCV (for image preprocessing)
- Scikit-learn (metrics & evaluation)

## 🧪 Features

- Image preprocessing (resizing, normalization, augmentation)
- CNN-based model training and evaluation
- Train-validation split and performance metrics (accuracy, loss, classification report)
- Visualization of training curves and predictions
- Support for 4 classes: **glioma**, **meningioma**, **pituitary**, and **no_tumor**

## 📊 Dataset

The dataset used is composed of MRI brain images and categorized into four types:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

> 📌 You can find the dataset at [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:

```bash
jupyter notebook "Brain Tumor Classification Using Deep Learning Algorithms.ipynb"
```

## ✅ Results

- Achieved training accuracy: ~98%
- Validation accuracy: ~95%
- Classification report shows strong F1-scores across all classes

## 🔍 Future Improvements

- Add model explainability with Grad-CAM
- Experiment with transfer learning (e.g., ResNet, VGG)
- Improve data augmentation strategies
- Deploy the model via Flask or Streamlit
