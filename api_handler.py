import requests

BASE_URL = "https://api.socialverseapp.com"
HEADERS = {
    "Flic-Token": "flic_b9c73e760ec8eae0b7468e7916e8a50a8a60ea7e862c32be44927f5a5ca69867"
}

def fetch_data(endpoint, params=None):
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, headers=HEADERS, params=params)
    
    # Log the response for debugging
    # print(f"Response status code: {response.status_code}")
    # print("Response content:", response.json())
    
    if response.status_code == 200:
        return response.json()  # Safely return 'data' if it exists
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}, Response: {response.text}")


# Fetch specific data
def get_viewed_posts():
    return fetch_data("posts/view", {"page": 1, "page_size": 1000})

def get_liked_posts():
    return fetch_data("posts/like", {"page": 1, "page_size": 5})

def get_ratings():
    return fetch_data("posts/rating", {"page": 1, "page_size": 5})

def get_all_posts():
    return fetch_data("posts/summary/get", {"page": 1, "page_size": 1000})

def get_all_users():
    return fetch_data("users/get_all", {"page": 1, "page_size": 1000})
