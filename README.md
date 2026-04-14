# 🩺 Pneumonia Detection using ResNet-18 (Transfer Learning)

## 📌 Project Overview

Pneumonia is a serious respiratory disease that requires early detection. In this project, a deep learning model is developed to classify chest X-ray images into:

* NORMAL
* PNEUMONIA

We use **Transfer Learning** with a pre-trained ResNet-18 model to achieve efficient and fast training.

---

## 🚀 Approach

### 1. Model

* Pre-trained **ResNet-18**
* All layers frozen except the final fully connected layer

### 2. Modification

* Final layer changed to binary classifier:

```python
resnet18.fc = nn.Linear(in_features, 1)
```

### 3. Training

* Loss Function: BCEWithLogitsLoss
* Optimizer: Adam
* Epochs: 3

---

## 📊 Results

| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.580 |
| F1 Score | 0.704 |

---

## 📂 Dataset

* Chest X-ray dataset
* 300 training images
* 100 testing images

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python train.py
```

---

## 🧠 Future Improvements

* Train for more epochs
* Add validation set
* Use data augmentation
* Try deeper models (ResNet-50, EfficientNet)

---

## 👨‍💻 Author

Sojib Chandra Roy
