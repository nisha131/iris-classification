
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the pre-trained model and scaler
df = pd.read_csv('iris_dataset.csv')
X = df.iloc[:, 2:6].values
y = df['Species'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LogisticRegression(random_state=42)
clf.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepalLength'])
    sepal_width = float(request.form['sepalWidth'])
    petal_length = float(request.form['petalLength'])
    petal_width = float(request.form['petalWidth'])

    # Scale the input values
    input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict the species
    prediction = clf.predict(input_data)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

