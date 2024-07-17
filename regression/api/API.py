from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the model serving API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Here you would add the prediction code
    return jsonify({'prediction': 'dummy_prediction'})

if __name__ == '__main__':
    app.run(debug=True)




'''
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Load your trained model
model = torch.load('model.pth')
model.eval()

def preprocess(input_data):
    input_tensor = torch.tensor(input_data)
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor

def postprocess(output_tensor):
    prediction = output_tensor.argmax(dim=1).item()
    return prediction

@app.route('/')
def home():
    return "Welcome to the model serving API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_tensor = preprocess(data['input'])
    with torch.no_grad():
        output = model(input_tensor)
    prediction = postprocess(output)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
'''