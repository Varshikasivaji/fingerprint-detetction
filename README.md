# fingerprint-detetction

CNN-based fingerprint image processing system for blood group classification
# Enhanced AI-Based Blood Group Detection Using Fingerprints

## 📌 Abstract
This project proposes a novel method to predict an individual’s blood group from fingerprint patterns using advanced image and signal processing techniques. High-resolution fingerprint images are captured and processed through histogram equalization, noise removal, ridge enhancement, and feature extraction. The extracted features are analyzed and classified using a Convolutional Neural Network (CNN) and ResNet architecture.  
The primary objective of the system is to provide a **non-invasive, rapid, and cost-effective alternative** to conventional blood group detection by identifying unique correlations between fingerprint characteristics and blood groups.

---

## 🧠 Technologies Used
- **Python**
- **CNN / ResNet**
- **OpenCV**
- **TensorFlow / Keras**
- **Deep Learning**
- **Image Processing Techniques**

---

## 📁 Project Structure
Enhanced-AI-BloodGroup-Detection/
│── src/
│ ├── preprocessing.py
│ ├── feature_extraction.py
│ ├── cnn_model.py
│ ├── resnet_classifier.py
│ └── inference.py

---

## 🌐 Dataset
A **limited number of fingerprint images** are uploaded directly to the repository in the `sample_dataset/` folder.  
These are included for demonstration and testing.

If you later plan to include a full dataset:
- Upload to Google Drive
- Add link here in README

---

## 🔍 Methodology / Proposed System
### ✔️ **1. Data Acquisition**
High-resolution fingerprint images are collected through scanners or mobile-based fingerprint capture systems.

### ✔️ **2. Preprocessing**
- Grayscale conversion  
- Histogram Equalization  
- Gaussian/Median Noise Filtering  
- Ridge & Edge Enhancement  
- Image Normalization  
- ROI detection & cropping  

### ✔️ **3. Feature Extraction**
- Minutiae Analysis  
- Ridge Orientation & Frequency  
- Texture Features  
- Gabor / Sobel / Canny filters  
- Deep feature extraction using pre-trained **ResNet**

### ✔️ **4. Classification**
A hybrid CNN–ResNet model is used:
- CNN for spatial fingerprint feature extraction  
- ResNet layers for deeper pattern learning  
- Softmax output for blood group prediction (A, B, AB, O, Rh±)

### ✔️ **5. Output**
Predicted blood group is displayed along with confidence score.

---

## 🧠 Model Architecture (Summary)
- **Input Layer**: 128×128 or 224×224 grayscale fingerprint  
- **Convolution Layers** (Feature extraction)  
- **Batch Normalization & ReLU Activation**  
- **MaxPooling Layers**  
- **ResNet Block Integration**  
- **Flatten Layer**  
- **Dense Layers**  
- **Softmax Layer** for classification  

---

## ▶️ How to Run the Project

### **1. Install Dependencies**

### **2. Run Preprocessing**

### **3. Train the Model**
or  

### **4. Test / Predict Blood Group**

---

## 📊 Results (Template – you can fill this later)
| Model | Accuracy | Dataset Size | Preprocessing | Notes |
|-------|----------|--------------|----------------|--------|
| CNN (Custom) | --% | -- images | Basic filtering | Initial testing |
| ResNet-50 | --% | -- images | Advanced preprocessing | Best performance |

Add your real values once you complete training.

---

## 📐 Block Diagram (Text Version)
Fingerprint Input
↓
Preprocessing
(Noise Filtering, Histogram Equalization, Ridge Enhancement)
↓
Feature Extraction
(Texture + Minutiae + Deep Features)
↓
CNN / ResNet Classification
↓
Blood Group Prediction (A, B, AB, O)

---

## 👨‍💻 Contributors
- **Your Name** (Lead Developer & Researcher)  
(Add more names if team members exist)

---

## 📜 License
This project is released under the **MIT License**.  
Users may use, modify, and distribute the code with proper attribution.

---

## 📢 Citation
If used for research, please cite:


---

## ⭐ Suggestions / Future Scope
- Integrating mobile-app based fingerprint scanning  
- Deploying model as a web API  
- Expanding dataset size for higher accuracy  
- Implementing real-time blood group prediction  
