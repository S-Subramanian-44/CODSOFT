# Movie Genre Prediction

## Introduction

This is a simple Flask web application for predicting the genre of a movie based on its description. It uses a machine learning model (Linear Support Vector Classifier) trained on a dataset of movie descriptions.

# Model Training
The machine learning model, a Linear Support Vector Classifier (LinearSVC), is trained on a dataset (train_data) containing movie descriptions and their corresponding genres. The training process involves learning the patterns and relationships between movie descriptions and genres.

# Prediction Process
## Description Preprocessing
- Input Description:
  Users input a movie description through the web interface.
- TF-IDF Transformation:
  The description undergoes TF-IDF transformation using the TfidfVectorizer. This step converts the raw text into a numerical representation.

# Genre Prediction
## Model Prediction:
The preprocessed description is fed into the trained model to predict a genre label.

# Inverse Label Transformation
## Label Decoding:
The predicted label is transformed back into the original genre using the LabelEncoder.

# Displaying the Prediction
## Output to User:
The predicted genre is returned as a JSON response and displayed on the web page.

This prediction process allows users to input a movie description, have it transformed into a format the model understands, and receive a predicted genre as the output.

# HTML Template
The HTML template (templates/index.html) is designed for a clean and user-friendly interface. It includes a textarea for entering movie descriptions and a button for initiating predictions.

# Let's Connect!
🌐 LinkedIn: https://www.linkedin.com/in/subramanian-s-ab94302a1/ 
📧 Email: subramanian160104@gmail.com
