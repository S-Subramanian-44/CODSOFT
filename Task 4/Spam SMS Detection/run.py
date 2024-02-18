from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        message = request.form["message"]
        prediction = predict_spam(message)
        return render_template("index.html", prediction=prediction)
    return render_template("index.html", prediction=None)

def train_model():
    # Step 1: Data Preprocessing
    data = pd.read_csv("spam.csv", encoding='ISO-8859-1')
    data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})  # Convert labels to numerical values

    # Step 2: Feature Engineering
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(data['v2'])
    y = data['v1']

    # Step 3: Model Selection
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Training
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Step 5: Evaluation
    y_pred = model.predict(X_test)
    print("Evaluation Report:")
    print(classification_report(y_test, y_pred))

    # Step 6: Save the trained model and TF-IDF vectorizer
    joblib.dump(model, "spam_classifier_model.joblib")
    joblib.dump(tfidf, "tfidf_vectorizer.joblib")

def predict_spam(input_message):
    # Load the trained model and TF-IDF vectorizer
    model = joblib.load("spam_classifier_model.joblib")
    tfidf = joblib.load("tfidf_vectorizer.joblib")

    input_message_tfidf = tfidf.transform([input_message])
    prediction = model.predict(input_message_tfidf)
    if prediction[0] == 0:
        return "Ham"
    else:
        return "Spam"

if __name__ == "__main__":
    train_model()  # Train the model when the script is executed
    app.run(debug=True)
