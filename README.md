# Brain Tumor Classification using Machine Learning

![image](https://github.com/user-attachments/assets/68d9a596-26de-42ec-ae59-e5b5f9f5ba3f)

## Project Overview
This project uses machine learning techniques to classify brain tumors based on MRI images. Leveraging supervised learning, the model predicts the type of tumor (glioma, meningioma, pituitary tumor, or no tumor) with high accuracy. A Random Forest Classifier is employed for predictions after preprocessing and dimensionality reduction.

---

## Dataset
The dataset used in this project consists of MRI images for training and testing, categorized into four classes:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

### Source:
- Dataset: [Kaggle Brain Tumor Classification MRI Dataset](https://www.kaggle.com).

---

## Steps Implemented

### 1. **Data Loading and Preprocessing**
- Imported **training** and **testing** image data from the specified paths.
- Resized all images to 300x300 pixels for uniformity.
- Stored images and corresponding class labels in arrays (`X_train` and `Y_train`).

### 2. **Data Augmentation**
- Preprocessed training and testing images to ensure consistency.
- Normalized pixel values to a scale of 0-1 for better training performance.

### 3. **Dimensionality Reduction**
- Reshaped the image data to reduce dimensionality using `Principal Component Analysis (PCA)` to improve accuracy and visualization.

### 4. **Model Building**
- Implemented a **Random Forest Classifier** to classify tumors.
- Training and testing datasets were split using an 80-20 ratio.

### 5. **Model Evaluation**
- The model achieved the following performance scores:
  - **Training Accuracy**: 100%
  - **Testing Accuracy**: 92.19%

---

## Key Features
- **Image Preprocessing**: Applied resizing, normalization, and reshaping techniques for efficient model training.
- **Random Forest Classifier**: Utilized an ensemble learning method for classification.
- **Dimensionality Reduction**: Used PCA to handle high-dimensional data and improve computational efficiency.

---

## Results
The model successfully classified brain tumors into four categories with a high testing accuracy of **92.19%**. Predictions for sample test images include:
- Predicted class for sample 9: `meningioma_tumor`
- Predicted class for sample 600: `pituitary_tumor`
- Predicted class for sample 50: `pituitary_tumor`
- Predicted class for sample 60: `no_tumor`

---

## Libraries Used
- **Data Manipulation and Analysis**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `tqdm`
- **Image Processing**: `cv2 (OpenCV)`
- **Machine Learning**: `sklearn` (PCA, Random Forest Classifier)
- **TensorFlow** (For shuffling and data handling)

---

## How to Use
1. **Setup Environment**:
   - Install dependencies:
     ```bash
     pip install pandas numpy matplotlib tqdm scikit-learn opencv-python
     ```
2. **Download Dataset**:
   - Ensure the training and testing data folders are properly structured.
3. **Run Script**:
   - Preprocess and load data.
   - Train the model with the provided script.
4. **View Results**:
   - Check model predictions and accuracy.

---

## Future Improvements
- Experiment with more advanced neural networks like CNNs for improved accuracy.
- Implement additional preprocessing techniques like image augmentation.
- Explore hyperparameter tuning for model optimization.
