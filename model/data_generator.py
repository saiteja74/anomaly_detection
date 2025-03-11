import random
import uuid
import time
import pandas as pd

def generate_fake_transaction_data(n=5000):
    data = []
    for _ in range(n):
        transaction_id = str(uuid.uuid4())
        sender = "0x" + ''.join(random.choices('abcdef0123456789', k=40))
        receiver = "0x" + ''.join(random.choices('abcdef0123456789', k=40))
        amount = round(random.uniform(0.001, 100), 4)
        timestamp = int(time.time()) - random.randint(1, 31536000)
        gas_fee = round(random.uniform(0.0001, 0.1), 6)
        transaction_count = random.randint(1, 500)
        wallet_age = random.randint(1, 365)
        # 10% Fraudulent Transactions
        is_fraud = 1 if (gas_fee > 0.05 and transaction_count > 300) or (amount > 80) else 0
        data.append([transaction_id, sender, receiver, amount, timestamp, gas_fee, transaction_count, wallet_age, is_fraud])
    df = pd.DataFrame(data, columns=["transaction_id", "sender_address", "receiver_address", "amount", "timestamp", "gas_fee", 
                                     "transaction_count", "wallet_age", "is_fraud"])
    df.to_csv("synthetic_transactions.csv", index=False)

generate_fake_transaction_data()
