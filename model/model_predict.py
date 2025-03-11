
import requests
import json

# Define the transaction data
transaction_data = {
    'amount': 50,
    'timestamp': 1643723400,
    'gas_fee': 0.06,
    'transaction_count': 350,
    'wallet_age': 200,
    'model': 'ocsvm'  # iso or ocsvm to select the model
}

# Set the API endpoint URL
url = 'http://localhost:5000/detect_fraud'

# Make the POST request
response = requests.post(url, json=transaction_data)

# Print the response
print(response.json())
