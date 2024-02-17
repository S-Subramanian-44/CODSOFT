# Customer Churn Prediction

## Introduction

This Flask web application is developed to predict customer churn for subscription-based services or businesses. The model utilizes historical customer data, incorporating features like usage behavior and customer demographics. Various machine learning algorithms, including Logistic Regression, Random Forests, and Gradient Boosting, are explored to predict churn patterns effectively.

### Dataset
Link to dataset: [Churn_Modelling.csv](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

## Model Training

The machine learning model, a Gradient Boosting Classifier, is trained on the provided dataset. The training process involves preprocessing the data, splitting it into training and testing sets, feature scaling, and finally training the model using Gradient Boosting.

## Prediction Process

### Data Preprocessing
- Irrelevant Column Removal:
  The dataset's irrelevant columns ('RowNumber', 'Surname') are dropped.
- Categorical Variable Handling:
  Categorical variables ('Geography', 'Gender') are encoded using LabelEncoder.

### Model Prediction
- Trained Model:
  The preprocessed data is fed into the trained Gradient Boosting Classifier to predict customer churn.

## Evaluation Metrics

Model performance is evaluated using key metrics:
- Accuracy
- Precision
- Recall
- F1 Score

## Displaying Results

The Flask application displays churn customers' details along with evaluation metrics on the web interface.

## HTML Template

The HTML template (templates/index.html) provides a user-friendly interface to view churn customers' details and evaluation metrics.

## Let's Connect!

🌐 LinkedIn: https://www.linkedin.com/in/subramanian-s-ab94302a1/ 
📧 Email: subramanian160104@gmail.com
