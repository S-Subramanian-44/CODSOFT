# Spam SMS Detection

## Introduction

This Flask web application is developed to classify SMS messages as spam or legitimate. The model utilize the technique of TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction and several classifiers. It aims to effectively identify spam messages based on their content.

### Dataset
Link to dataset: [spam.csv](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## Model Training

The machine learning model is trained on the provided dataset. The training process involves preprocessing the text data, extracting features using TF-IDF, splitting the dataset into training and testing sets, and training various classifiers including Naive Bayes, Logistic Regression, and Support Vector Machines.

## Prediction Process

### Data Preprocessing
- Text Preprocessing:
  Text data is preprocessed by removing punctuation, converting text to lowercase, and tokenizing the text into individual words.
- Feature Extraction:
  TF-IDF vectorization is applied to convert text data into numerical features.

### Model Prediction
- Trained Models:
  The preprocessed data is fed into the trained classifiers to predict whether a given SMS message is spam or legitimate.

## Evaluation Metrics

Model performance is evaluated using standard metrics:
- Accuracy
- Precision
- Recall
- F1 Score

## Displaying Results

The Flask application displays the prediction results for user-input SMS messages along with evaluation metrics on the web interface.

## HTML Template

The HTML template provides a user-friendly interface to input SMS messages and view the prediction results and evaluation metrics.

## Let's Connect!

🌐 LinkedIn: https://www.linkedin.com/in/subramanian-s-ab94302a1/ 
📧 Email: subramanian160104@gmail.com
