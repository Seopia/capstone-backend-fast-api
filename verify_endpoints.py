import requests
import jwt
import datetime
import os
import sys
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8000"
JWT_SECRET = os.getenv("JWT_SECRET", "1234") # Fallback for test

def create_token(user_code=1):
    payload = {
        "userCode": user_code,
        "user_code": user_code, # Sending both to be safe
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return token

def test_chat():
    print("Testing /chat endpoint...")
    token = create_token()
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": "Hello, this is a test.", "convId": "507f1f77bcf86cd799439011", "isJailbreak": False} # Valid ObjectId format for mocking
    
    try:
        response = requests.post(f"{BASE_URL}/chat", headers=headers, json=data)
        response.raise_for_status()
        print(f"Chat Response: {response.text[:200]}...")
        if response.status_code == 200:
            print("Chat Test PASSED")
        else:
            print("Chat Test FAILED")
    except Exception as e:
        print(f"Chat Test FAILED with error: {e}")
        # Print detailed response if available
        if 'response' in locals() and response is not None:
             print(response.text)

def test_analyze():
    print("Testing /analyze endpoint...")
    token = create_token()
    headers = {"Authorization": f"Bearer {token}"}
    data = {"convId": "507f1f77bcf86cd799439011"}
    
    try:
        response = requests.post(f"{BASE_URL}/analyze", headers=headers, json=data)
        response.raise_for_status()
        print(f"Analyze Response: {response.json()}")
        print("Analyze Test PASSED")
    except Exception as e:
        # 404/500 might happen if no data, but server shouldn't crash
        print(f"Analyze Test ended with: {e}")
        if 'response' in locals() and response is not None:
             print(response.text)

def main():
    print("Starting Verification...")
    test_chat()
    # analyze might depend on chat history existing
    test_analyze()
    print("Verification Finished.")

if __name__ == "__main__":
    main()
