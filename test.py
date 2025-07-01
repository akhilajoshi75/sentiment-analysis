import requests

url = "http://localhost:5000/predict"
payload = {"text": "I love this project!"}
response = requests.post(url, json=payload)
print(response.json())
