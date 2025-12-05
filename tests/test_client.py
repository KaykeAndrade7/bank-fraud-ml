from clients.python.client import FraudClient

client = FraudClient("http://127.0.0.1:8000")

print(client.healthcheck())

exemplo = {
    "Time": 1000,
    "V1": -1.2, "V2": 0.4, "V3": 0.9, "V4": -0.7, "V5": 1.5,
    "V6": -0.1, "V7": 0.3, "V8": 0.2, "V9": 0.0, "V10": -0.9,
    "V11": 0.1, "V12": 0.6, "V13": -1.1, "V14": 0.7, "V15": 0.3,
    "V16": 0.2, "V17": -0.4, "V18": 0.8, "V19": -0.2, "V20": 0.0,
    "V21": 0.5, "V22": -0.3, "V23": 0.6, "V24": -0.1, "V25": 0.2,
    "V26": -0.5, "V27": 0.7, "V28": -0.6,
    "Amount": 120.55
}

print(client.predict_single(exemplo))
