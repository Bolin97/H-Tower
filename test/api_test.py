import json
import random

import requests


question = '''process_info = processing_list[infer_token]
process_info 结构如下
class PROCESSINGTOKEN:
    def __init__(self):
        self.user_value = None
        self.infer_value = None
        self.value_type = None
        self.dialog_history = []
        self.options = []
        self.unsuccessful = 0  # 推理失败的次数
        self.token_answer_type = None

如果process_info 的user_value 修改了，是否会影响processing_list'''

url = "https://oa.api2d.net/v1/chat/completions"
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
headers = {
    'Authorization': 'Bearer fk208078-Z9HS0S4q3UkYDOo8WxsEp3EK4rg0YuGc',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
}
response = requests.request("POST", url, headers=headers, data=payload)
res = response.json()
res = res["choices"][0]["message"]["content"]
print(res)