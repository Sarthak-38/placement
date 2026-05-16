from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('placement_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    cgpa = float(request.form['cgpa'])
    internships = int(request.form['internships'])

    features = np.array([[cgpa, internships]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Student is likely to be PLACED"
    else:
        result = "Student is likely to NOT be placed"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)