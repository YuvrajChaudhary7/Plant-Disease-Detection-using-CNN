# 🌿 Plant Disease Detection using Deep Convolutional Neural Networks (CNN)

## 📌 Project Overview
Plant diseases significantly affect agricultural productivity and crop quality. Early detection of plant diseases helps farmers take preventive measures and reduce crop losses.

This project implements an AI-based plant disease detection system using **Convolutional Neural Networks (CNN)**. The system analyzes images of tomato leaves and classifies them into different disease categories.

The trained model is integrated into a **Flask web application**, allowing users to upload leaf images and receive disease predictions along with confidence scores and treatment suggestions.

---

## 🎯 Objectives
- Develop an automated system for detecting plant diseases from leaf images.
- Apply deep learning techniques for image classification.
- Build a user-friendly web interface for disease prediction.
- Provide confidence scores and treatment recommendations.

---

## 🧠 Technologies Used

| Technology | Purpose |
|------------|--------|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Flask | Web application framework |
| OpenCV | Image preprocessing |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| HTML/CSS | Frontend interface |

---

## 📂 Project Structure

```
plant-disease-detection
│
├── dataset
│   ├── Tomato_Early_blight
│   ├── Tomato_Late_blight
│   └── Tomato_healthy
│
├── model
│   └── model.pth
│
├── uploads
│
├── templates
│   ├── index.html
│   └── result.html
│
├── static
│   ├── css
│   └── images
│
├── train_model.py
├── predict.py
├── app.py
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset
The model is trained using the **PlantVillage dataset**, which contains labeled images of plant leaves with different diseases.

### Classes Used
- Tomato Early Blight
- Tomato Late Blight
- Tomato Healthy

Dataset Source  
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Run the Flask web application:

```bash
python app.py
```

The browser will automatically open at:

```
http://127.0.0.1:5000
```

---

## 🖼️ Application Workflow

1. User uploads a tomato leaf image
2. Image is preprocessed (resize and normalization)
3. CNN model analyzes the image
4. Disease prediction is generated
5. System displays:
   - Disease name
   - Confidence percentage
   - Treatment suggestion

---

## 🧬 Model Architecture

The CNN model consists of the following layers:

- Convolution layers
- ReLU activation
- Max pooling layers
- Fully connected layers
- Softmax output layer

These layers enable the model to extract important visual features from leaf images and classify diseases effectively.

---

## 📈 Model Performance

| Metric | Value |
|------|------|
| Accuracy | ~91% |
| Precision | ~90% |
| Recall | ~89% |
| F1 Score | ~90% |

---

## ⚠️ Limitations

- The model is trained on a limited number of disease classes.
- Non-leaf images may still be classified as one of the known diseases.
- Performance depends on lighting conditions and image quality.

---

## 🔮 Future Improvements

- Add more plant disease classes
- Develop a mobile application
- Enable real-time detection using smartphone cameras
- Improve accuracy using advanced architectures such as ResNet or EfficientNet
- Deploy the system on cloud platforms
- Implementing the system on drone systems

---

## 👨‍💻 Authors

**Yuvraj**  
UID: 25LBCS3172  

**Nishant Kumar**  
UID: 25LBCS3201  

**Aditya Singh**  
UID: 25LBCS3146  

---

## 📚 References

1. Pandian, J. Arun et al. *Plant Disease Detection Using Deep Convolutional Neural Network*.
2. IEEE Research Paper – *Plant Disease Detection Using CNN*.
3. IEEE Research Paper – *Plant Disease Detection Using Image Preprocessing Techniques*.
4. ScienceDirect – *Plant Disease Detection Using Machine Learning*.
5. ScienceDirect – *Plant Disease Detection Using CNN*.

---

## ⭐ Acknowledgement

This project was developed as part of the **Computer Ecosystem course project** in the **BTech CSE (AI & ML) program**.
