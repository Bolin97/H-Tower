import requests
from urllib.parse import quote
import json
user_msg, history = "输入的指令是:'''请查询房龄5-10年,商品房住宅,350-400万,顶层,豪华装修的二手房信息'''\n请抽取#面积#\n如果info的提供了候选值,那就请从中选取," \
                    "info如下\n'''['不限', '50m以下', '50-60m', '60-70m', '70-80m', '80-90m', '90-100m', '100-120m', " \
                    "'120-140m', '140-210m', '210m以上']'''", []
# 对 user_msg 进行 URL 编码
encoded_user_msg = quote(user_msg)

resp = requests.post(f"http://127.0.0.1:8081/predict?user_msg={encoded_user_msg}", json=history)
if resp.status_code == 200:

    response, history = resp.json()["response"], resp.json()["history"]
    data_list = json.loads(response.replace("'", "\""))
    value = data_list[0]['value']
    print(value)
    # print(type(response))
