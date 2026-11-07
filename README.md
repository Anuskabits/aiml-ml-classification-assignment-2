
# Bank Marketing Subscription Prediction

A machine learning classification project comparing five algorithms to predict whether a bank customer will subscribe to a term deposit.

## 1. Problem Statement

The objective of this project is to develop and compare five machine learning classification models to predict whether a bank customer will subscribe to a term deposit based on customer and campaign-related information.

The five classification algorithms implemented are:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 2. Dataset Description

**Dataset:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

| Parameter | Description |
|---|---|
| Total records | 45,211 |
| Original features | 16 |
| Target variable | y |
| Classification type | Binary classification |
| Input features used | 15 |

The target variable `y` has two classes:

- `yes`: Customer subscribed to a term deposit.
- `no`: Customer did not subscribe to a term deposit.

The `duration` feature was excluded because it represents the duration of the phone call and would not be available before making the call. Therefore, 15 input features were used.

## 3. Data Preprocessing

The following preprocessing operations were performed:

- Checked duplicate records and missing values.
- Retained unknown categorical values as a separate category.
- Converted the target variable into binary format (Yes = 1, No = 0).
- Applied median imputation and StandardScaler to numerical features.
- Applied most-frequent imputation and OneHotEncoder to categorical features.
- Used an 80:20 stratified train-test split.
- Used random_state = 42 for reproducibility.

**Training records:** 36,168  
**Testing records:** 9,043

## 4. Model Evaluation

All five classification models were trained using the same training dataset and evaluated using the same test dataset.

The following six evaluation metrics were calculated:

1. Accuracy
2. AUC
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)

### Model Comparison Results

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7548 | 0.7722 | 0.2662 | 0.6238 | 0.3732 | 0.2853 |
| Decision Tree | 0.8402 | 0.6008 | 0.3059 | 0.2883 | 0.2968 | 0.2069 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| SVM | 0.7578 | 0.7718 | 0.2684 | 0.6200 | 0.3746 | 0.2867 |

## 5. Model Observations

### Logistic Regression

Achieved the highest AUC of 0.7722, indicating good ability to distinguish between the two classes across different classification thresholds. It also achieved a recall of 0.6238.

### Decision Tree

Achieved an accuracy of 0.8402 but had a relatively low AUC of 0.6008 and F1 Score of 0.2968.

### K-Nearest Neighbors (KNN)

Achieved the highest accuracy of 0.8874 and precision of 0.5529. However, its recall was only 0.1975, indicating that it missed many actual subscribers.

### Gaussian Naive Bayes

Achieved the highest F1 Score of 0.4123 and MCC of 0.3271. It provided the best balance between precision and recall among the five models.

### Support Vector Machine (SVM)

Achieved an AUC of 0.7718 and recall of 0.6200. Its performance was similar to Logistic Regression, although its precision was relatively low.

## 6. Overall Conclusion

Naive Bayes was the best-performing model based on F1 Score and MCC.

Although KNN achieved the highest accuracy, it had the lowest recall. This demonstrates why accuracy alone is not sufficient for evaluating models on an imbalanced dataset.

Naive Bayes provided the best overall balance between precision and recall for this classification task.

## 7. Streamlit Application

The Streamlit application provides the following features:

- Selection of any of the five trained classification models.
- Upload of a test CSV dataset.
- Display of all six evaluation metrics.
- Display of the confusion matrix.
- Display of the classification report.

The application loads the previously saved trained models and does not perform model retraining.

**Live Application:** To be added after deployment.

## 8. GitHub Repository

**Repository Link:**

https://github.com/Anuskabits/aiml-ml-classification-assignment-2

## 9. Project Structure

```text
aiml-ml-classification-assignment-2/
│
├── ML_Assignment_2.ipynb
├── app.py
├── README.md
├── requirements.txt
├── bank-full.csv
├── test_data.csv
├── model_comparison.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── svm.pkl
```
