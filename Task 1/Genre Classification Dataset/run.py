from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.config['STATIC_FOLDER'] = static_folder

# Load training data
train_data = pd.read_csv("train_data.txt", sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python')

# Fill missing values in 'DESCRIPTION' column
train_data['DESCRIPTION'] = train_data['DESCRIPTION'].fillna("")

# Create TF-IDF vectorizer
t_v = TfidfVectorizer(stop_words='english', max_features=100000)

# Transform the training data descriptions
X_train = t_v.fit_transform(train_data['DESCRIPTION'])

# Encode the target variable 'GENRE'
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['GENRE'])

# Split the training data into training and validation sets
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create LinearSVC classifier and train it on the training subset
clf = LinearSVC(dual=False)
clf.fit(X_train_sub, y_train_sub)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the prediction
@app.route('/predict_movie', methods=['POST'])
def predict_movie_route():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        # Extract the 'description' from the JSON data
        description = data['description']
        # Make the prediction using the predict_movie function
        t_v1 = t_v.transform([description])
        pred_label = clf.predict(t_v1)
        prediction = label_encoder.inverse_transform(pred_label)[0]
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction})
    except Exception as e:
        # Return an error message if something goes wrong
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)
