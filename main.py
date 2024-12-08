from flask import Flask, render_template, request
import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np

MODEL_PATH = './model/neural_network_model.pth'
PREPROCESSOR_PATH = './model/preprocessor.pkl'

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def load_model():
    model = NeuralNetwork(input_size=84) 
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        print("Received POST request with form data:", request.form)

        model = load_model()
        preprocessor = load_preprocessor()

        input_data = {
            'Gender': request.form['gender'],
            'Age': int(request.form['age']),
            'Work Pressure': float(request.form['work_pressure']),
            'Job Satisfaction': float(request.form['job_satisfaction']),
            'Sleep Duration': request.form['sleep_duration'],
            'Dietary Habits': request.form['dietary_habits'],
            'Have you ever had suicidal thoughts ?': request.form['suicidal_thoughts'],
            'Work Hours': int(request.form['work_hours']),
            'Financial Stress': int(request.form['financial_stress']),
            'Family History of Mental Illness': request.form['family_history']
        }

        print("Input data:", input_data)

        input_df = pd.DataFrame([input_data])
        processed_input = preprocessor.transform(input_df)

        if hasattr(processed_input, "toarray"):
            processed_input = processed_input.toarray()

        input_tensor = torch.tensor(processed_input, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(input_tensor)
            prediction_label = "Depressed" if prediction.item() > 0.5 else "Not Depressed"
        
        print("Prediction:", prediction_label)

        return render_template('result.html', prediction=prediction_label)
    except Exception as e:
        print("Error occurred:", str(e))
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
