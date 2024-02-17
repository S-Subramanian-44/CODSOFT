from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

def preprocess_data(data):
    # Drop irrelevant columns
    data.drop(['RowNumber', 'Surname'], axis=1, inplace=True)

    # Handle categorical variables
    label_encoder = LabelEncoder()
    data['Geography'] = label_encoder.fit_transform(data['Geography'])
    data['Gender'] = label_encoder.fit_transform(data['Gender'])

    return data

@app.route('/')
def churn_customers():
    # Load the dataset
    data = pd.read_csv('Churn_Modelling.csv')

    # Preprocess data
    data = preprocess_data(data)

    # Split features and target variable
    X = data.drop(['CustomerId', 'Exited'], axis=1)
    y = data['Exited']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Get churn customers' indices based on predictions
    churn_indices = y_test[y_pred == 1].index

    # Display churn customers' details
    churn_customers_details = data.loc[churn_indices]

    return render_template('index.html', churn_customers=churn_customers_details.to_html(index=False), accuracy=accuracy, precision=precision, recall=recall, f1=f1)

if __name__ == '__main__':
    app.run(debug=True)
