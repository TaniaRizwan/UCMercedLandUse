# 🌍 Aerial Image Classification with CNNs (UC Merced Dataset)

This project showcases classifying high-resolution aerial images into land-use categories using deep learning. Leveraging **transfer learning with the InceptionResNetV2** architecture, the model accurately identifies 21 land-use classes, achieving over **92% accuracy**.

This was built as a final project for ENEL 525: Machine Learning for Engineers, at the University of Calgary.

---

## 📦 Dataset

The model is trained on the [UC Merced Land Use Dataset](https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset). This contains:

- 2,100 RGB images
- 21 classes (e.g. Forest, Beach, Freeway, Residential)
- Image resolution: 256 × 256 pixels 

---

## 🧠 Project Overview

- **Model Architecture**: Transfer learning with [InceptionResNetV2](https://arxiv.org/abs/1602.07261) + custom classification head
- **Preprocessing**: Resizing, RGB conversion, pixel normalization
- **Data Augmentation**:
  - Horizontal flipping
  - Rotation (±20%)
  - Zoom (±20%)
  - Contrast adjustment
  - Translation (±10%)
- **Training Setup**:
  - 70% training, 15% validation, 15% testing
  - Callbacks include early stopping & dynamic learning rate adjustment
  - Trained for 20 epochs with Adam optimizer (`lr=1e-4`)

---

## 📈 Performance
The model's performance was evaluated using the following metrics:
| Metric     | Training | Validation | Testing |
|------------|----------|------------|---------|
| Accuracy   | 89%      | 91%        | **92%** |
| F1 Score   | 88%      | 91%        | **92%** |
| Precision  | 93%      | 94%        | **95%** |
| Recall     | 85%      | 88%        | **89%** |
| Loss       | 0.34     | 0.29       | **0.29** |

**Result**: The model demonstrated strong generalization, low overfitting, and high accuracy across the complex multi-class dataset.

**Confusion Matrix & Classification Report**: Included to analyze misclassifications between visually similar classes like "Dense Residential" vs "Buildings". 

---

## 🏗️ Model Architecture

```
Input Image (256x256x3)
   ↓
InceptionResNetV2 (Frozen convolutional base)
   ↓
GlobalAveragePooling2D
   ↓
Dense(512, ReLU)
   ↓
Dropout(0.5)
   ↓
Dense(21, Softmax)
```

---

## 💡 Key Learnings

- Learned to evaluate effective **transfer learning** approaches and apply to a real-world problem
- Improved model generalization through **augmentation and regularization**
- Gained hands-on experience tuning hyperparameters and interpreting model performance

---

## 👩‍💻 Author

**Tania Rizwan**  
Biomedical Engineering and CPSC Student @ University of Calgary
📫 [LinkedIn](https://www.linkedin.com/in/taniarizwan/) 

---

## 📝 License

This project is open for educational and personal use.  
Please credit the author if reusing any part of the work.
