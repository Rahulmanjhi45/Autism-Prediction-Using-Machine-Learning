# Autism Prediction Using Machine Learning

This project predicts the likelihood of Autism Spectrum Disorder (ASD) using machine learning models, employing behavioral, demographic, and other data to achieve accurate predictions. It leverages feature engineering, data preprocessing, and model optimization techniques to develop a reliable prediction system

## Table of Contents

Overview

Dataset

Exploratory Data Analysis

Data Preprocessing

Model Training

Evaluation

Usage

Dependencies

Acknowledgments


## Overview

The objective of this project is to identify individuals at risk of Autism Spectrum Disorder by analyzing behavioral and demographic features. The process involves:

1. Preprocessing the data to handle missing values and imbalances.

2. Building and tuning machine learning models.

3. Selecting the best-performing model based on evaluation metrics.


##Dataset

The dataset consists of 800 samples with 20 attributes after preprocessing. Key attributes include:

1. Behavioral scores (A1_Score to A10_Score)

2. Age, gender, ethnicity, and other demographics

3. Target variable: Class/ASD (1: ASD, 0: non-ASD)

## Highlights

1. The data is imbalanced, with a higher proportion of non-ASD cases.

2. Missing or ambiguous values in columns such as ethnicity and relation were handled.

## Exploratory Data Analysis

EDA was conducted to:

1. Understand data distribution and relationships between variables.

2. Detect and handle outliers using IQR.

3. Visualize distributions and correlations to identify trends.

### Key insights:

1. Behavioral scores follow a binary distribution.

2. Strong correlations exist between certain features and the target variable.

## Data Preprocessing

Steps:

1. Data Cleaning: Addressed missing and ambiguous values.

2. Outlier Handling: Replaced outliers in age and result columns with median values.

3. Feature Encoding: Categorical features were label-encoded.

4. Data Balancing: Applied SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance.

## Model Training

### Models Evaluated:

1. Decision Tree

2. Random Forest

3. XGBoost

## Hyperparameter Tuning

RandomizedSearchCV was employed to tune model hyperparameters.

## Best Model

#### Random Forest

1. Accuracy: 93% (Cross-validation)

2. Hyperparameters:     max_depth=20, n_estimators=50, bootstrap=False

## Evaluation

The best model was tested on a held-out test set:

#### Accuracy: 81.9%

#### Precision:

1. ASD: 59%

1. Non-ASD: 89%

#### Recall:

1. ASD: 64%

2. Non-ASD: 87%

#### F1-Score:

1. ASD: 61%

2. Non-ASD: 88%

## Usage

### Steps:

1. Install dependencies.

2. Load and preprocess the dataset.

3. Train models or use the pretrained model (best_model.pkl).

### Example
    import pickle
    from sklearn.metrics import classification_report
    
    # Load model
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    print(classification_report(y_test, y_pred))

## Dependencies

Python 3.8+

NumPy

Pandas

Scikit-learn

Seaborn

Matplotlib

XGBoost

Imbalanced-learn

## Acknowledgments

This project was developed by Rahul Manjhi. Special thanks to all contributors and the open-source community for providing tools and libraries that made this work possible.
