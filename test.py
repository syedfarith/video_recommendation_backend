import requests

url = "http://127.0.0.1:10000/recommendations/11"
params =None

response = requests.get(url, params=params)
if response.status_code == 200:
    print("Recommendations:", response.json())
else:
    print("Error:", response.status_code, response.text)
