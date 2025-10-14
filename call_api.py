
import requests

data = {
    "ph": 9.0,
    "Hardness": 204.89,
    "Solids": 20791.31,
    "Chloramines": 7.30,
    "Sulfate": 368.51,
    "Conductivity": 564.30,
    "Organic_carbon": 10.37,
    "Trihalomethanes": 86.99,
    "Turbidity": 2.96
}

response = requests.post("http://127.0.0.1:8000/prediccion", json=data)
print("Status:", response.status_code)
print("Respuesta:", response.json())