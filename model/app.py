


from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

app = Flask(__name__)

# Load models
if_model = IsolationForest(contamination=0.1, random_state=42)
ocsvm_model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)

# Load data
df = pd.read_csv("synthetic_transactions.csv")
X = df[['amount', 'timestamp', 'gas_fee', 'transaction_count', 'wallet_age']]

# Fit models
if_model.fit(X)
ocsvm_model.fit(X)

# POST endpoint for fraud detection
@app.route('/detect_fraud', methods=['POST'])
def detect_fraud():
    data = request.get_json()
    model_name = data.get('model', 'if')  # Default to Isolation Forest if not specified
    
    transaction_data = pd.DataFrame([[
        data['amount'],
        data['timestamp'],
        data['gas_fee'],
        data['transaction_count'],
        data['wallet_age']
    ]], columns=['amount', 'timestamp', 'gas_fee', 'transaction_count', 'wallet_age'])
    
    if model_name == 'iso':
        prediction = if_model.predict(transaction_data)
        fraud_probability = 1 if prediction[0] == -1 else 0
        model_used = 'Isolation Forest'
    elif model_name == 'ocsvm':
        prediction = ocsvm_model.predict(transaction_data)
        fraud_probability = 1 if prediction[0] == -1 else 0
        model_used = 'One-Class SVM'
    else:
        return jsonify({'error': 'Invalid model name'}), 400
    
    return jsonify({
        'model_used': model_used,
        'fraud_probability': fraud_probability
    })

# GET endpoint for model status
@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)
