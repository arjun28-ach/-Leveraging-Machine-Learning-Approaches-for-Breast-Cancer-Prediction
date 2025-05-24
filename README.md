# -Leveraging-Machine-Learning-Approaches-for-Breast-Cancer-Prediction

# Breast Cancer Prediction using Machine Learning

## MSc in Artificial Intelligence - Practical Skills Assessment
**Author:** Arjun Acharya
**Date:** May 2025 
**Module:** Introduction to Artificial Intelligence


## 1. Project Overview

This project aims to leverage machine learning techniques to predict whether a breast tumor is malignant or benign based on diagnostic measurements. The primary goal is to explore various classification algorithms, evaluate their performance, and identify the most effective model for this diagnostic task. This project was undertaken as part of the Practical Skills Assessment for the "Introduction to Artificial Intelligence" module.

Key objectives include:
*   Performing comprehensive Exploratory Data Analysis (EDA).
*   Preprocessing the data for optimal model training.
*   Training and evaluating multiple classification models:
    *   Logistic Regression
    *   Support Vector Machine (SVM)
    *   Decision Tree
    *   Random Forest
    *   K-Nearest Neighbors (KNN)
*   Optimizing the hyperparameters of the best-performing model.
*   Analyzing feature importance and discussing model interpretability.

---

## 2. Dataset

The dataset used is the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.
*   **Source:** The `breast_cancer_dataset.csv` file provided for the assignment, originally from the `https://github.com/abdelDebug/Breast-Cancer-Dataset/`)
*   **Description:** Contains 569 samples and 32 columns. The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
    *   `id`: Patient ID (removed before modeling)
    *   `diagnosis`: The target variable (M = malignant, B = benign)
    *   30 real-valued input features (e.g., `radius_mean`, `texture_mean`, `perimeter_mean`, etc.)

---

## 3. Methodology

The project follows a standard machine learning workflow:

1.  **Data Exploration (Task 1):**
    *   Loading the dataset.
    *   Investigating target variable distribution.
    *   Generating descriptive statistics for numerical features.
    *   Creating visualizations (histograms, box plots) to identify outliers and skew.
    *   Calculating correlation coefficients between features and the target.
2.  **Data Preparation (Task 2):**
    *   Handling the 'id' column (removing it).
    *   Encoding the categorical target variable ('diagnosis') into numerical format.
    *   Scaling numerical features using `StandardScaler`.
    *   Checking for and handling missing values (though none were present in this specific dataset version).
3.  **Model Training (Task 3):**
    *   Splitting data into training (80%) and testing (20%) sets with stratification.
    *   Training five classification models with their default hyperparameters.
4.  **Model Evaluation and Visualization (Task 4):**
    *   Evaluating models on the test set using Accuracy, Precision, Recall, F1-score, and AUC.
    *   Visualizing results (performance comparison, confusion matrix, ROC curves, feature importance, CV scores, precision-recall curve).
    *   Selecting the top-performing model for hyperparameter tuning using `GridSearchCV`.
    *   Comparing pre-tuning vs. post-tuning performance.
5.  **Conclusion and Future Work (Task 5):**
    *   Summarizing findings and discussing clinical implications and limitations.
    *   Suggesting avenues for future research and model improvement.

---


## 4. Requirements

The project requires Python 3.x and the following libraries:
*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn

A `requirements.txt` file can be generated using:
```bash
pip freeze > requirements.txt
(If you provide a requirements.txt, ensure it's accurate.)
