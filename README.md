# 🌿 Plant Disease Detection using Deep Learning

## 📌 Overview

This project is a deep learning-based web application that detects plant diseases from leaf images and visualizes model attention using Grad-CAM. It combines transfer learning, image preprocessing, and explainable AI to build an end-to-end intelligent system.

---

## 🚀 Features

* 🌱 Plant disease classification using deep learning
* 🔍 Transfer Learning with MobileNetV2
* 🧠 Explainable AI using Grad-CAM
* 🎨 Data augmentation for improved generalization
* 🌐 Interactive web app using Streamlit
* ⚡ Real-time predictions with confidence score

---

## 🧠 Model Details

### 🔹 Models Used

* Custom CNN (baseline)
* MobileNetV2 (transfer learning)

### 🔹 Best Model

* MobileNetV2 achieved ~95.5% validation accuracy

### 🔹 Why MobileNetV2?

* Lightweight and efficient
* High accuracy with low computation cost
* Suitable for deployment

---

## 📊 Dataset

* Dataset: PlantVillage Dataset
* Total classes: 38
* Images: ~54,000+
* Categories include:

  * Tomato diseases
  * Potato diseases
  * Apple diseases
  * Healthy leaves

---

## 🔧 Training Details

### ⚙️ Platform

* Training performed on **Google Colab (GPU)**

### 🔹 Techniques Used

* Data Augmentation:

  * Rotation
  * Zoom
  * Horizontal Flip
* EarlyStopping
* ReduceLROnPlateau

### 🔹 Preprocessing

* Images resized to 224×224
* Normalization (rescale = 1/255)

---

## 📓 Training Notebook

The complete training workflow is available in:

```text
notebook/plantdisease.ipynb
```

This notebook includes:

* Data loading and preprocessing
* Model building (CNN & MobileNetV2)
* Training on Google Colab GPU
* Performance evaluation
* Accuracy comparison plots

---

## 🔥 Grad-CAM Visualization

Grad-CAM is used to highlight the regions of the leaf image that influence the model’s prediction.

This improves:

* Interpretability
* Trust in predictions

---

## 🌐 Streamlit Web App

### Features:

* Upload leaf image
* Get prediction + confidence score
* Visualize Grad-CAM heatmap

### Run locally:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
plant-disease-detection/
│
├── models/
│   └── best_model.h5
│
├── reports/
│   ├── results.json
│   └── accuracy_plot.png
│
├── src/
│   └── predict.py
│
├── notebook/
│   └── plantdisease.ipynb
│
├── testimages/
├── app.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run prediction:

```bash
python src/predict.py
```

### Run web app:

```bash
streamlit run app.py
```

---

## ⚠️ Limitations

* Model trained on controlled dataset (PlantVillage)
* May not generalize perfectly to real-world images with complex backgrounds

---

## 🚀 Future Improvements

* Train on real-world dataset
* Add more robust augmentation
* Deploy on cloud for public access
* Improve model generalization

---

## 💡 Key Learnings

* Transfer learning significantly improves performance
* Data augmentation reduces overfitting
* Explainable AI (Grad-CAM) adds interpretability
* Deployment bridges gap between ML and real-world usage

---

## 👨‍💻 Author

* Praneeth Sangnal

---

## ⭐ Acknowledgements

* PlantVillage Dataset
* TensorFlow & Keras
* Streamlit
