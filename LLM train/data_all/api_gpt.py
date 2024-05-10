import json

import requests
import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.post("/process_messages/")
async def process_messages(question):
    headers = {
        'Authorization': 'Bearer fk208078-Z9HS0S4q3UkYDOo8WxsEp3EK4rg0YuGc',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    messages = [
        {
            'role': 'user',
            'content': question,
        }]
    payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "safe_mode": False
    })
    url = "https://oa.api2d.net/v1/chat/completions"
    response = requests.post(url, headers=headers, data=payload)
    res = response.json()
    return res["choices"][0]["message"]["content"]


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8081, workers=1)