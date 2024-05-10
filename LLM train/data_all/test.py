import json
import random

with open("tree_demand.json", "r", encoding="utf-8") as f:
    data = json.load(f)
trees = {}
for tree in data:
    page_source = tree["page_source"]
    trees[page_source] = tree["content"]
# %%
with open("re.json", "r", encoding="utf-8") as file_0:
    demand = json.load(file_0)
contexts = {}
purposes = {}
for data in demand:
    page_source = data['page_source']
    purpose = data['purpose']
    datas = []
    for inputs in data['data']:
        datas.append(inputs)
    contexts[page_source] = datas
    purposes[page_source] = purpose


# %%
def generate_demand(info, tree):
    result = ""
    for key, value in info.items():
        for item in tree:
            if key == item['key']:
                result = result + str(key) + "(" + item['annotation'] + ")" + ":" + str(value) + "\n"
    return result


import requests
from urllib.parse import quote


def gpt(message, tree, purpose):
    # 对 user_msg 进行 URL 编码
    prompt = ''''首先你对需求列表中的键值进行一些联想(think step by step)，将需求列表中具体的值改写成描述性的话或者同义词，然后再回答客服的问题。回答语气不要太正式，需求描述应该符合中文口语习惯。
##
目的:租房子
客服问题:您好，请问您对想要租住的房屋的房间数量有什么要求呢？
需求列表:
户型(房屋的房间数量或居室规模):三室
装修(房屋装饰和装潢的各种程度和类型):精装修
售价(房子的价格):1000万-2000万
think step by step:
三室的户型房间数量属于中等，因此\'''户型为三室\'''，可以用\'''房间数量中等\'''来替代
精装修的房子也就是装修非常精美，因此\'''装修为精装修\'''，可以用\'''装修精美\'''来替代
1000万-2000万的售价也就是1000多万，因此\'''售价为1000万-2000万\''',可以用\'''价格1000多万\'''来替代
user:我觉得房间数量中等就好，不用太多。但是我希望装修可以精美一些，价格嘛，1000多万吧。

##
目的:买二手车
客服问题:您好，请问您对车辆的价格有什么需求吗，或者说车的车型，车的特点等等，如果您有别的需求也请告诉我
需求列表:
能源(车辆的能源类型):纯电动
think step by step:
纯电动的车不会排放尾气，比较环保，因此\'''能源为纯电动\'''，可以用\'''环保\'''来替代
user:我希望能找到一辆环保的，对环境友好的车。

##
目的:看电影
客服问题:您对电影类型有什么偏好呢？
需求列表:
类型（影视作品的类型）:机甲
年龄范围（适合观看的年龄范围）:0-2岁
付费类型（影视作品是否需要付费）:免费
think step by step:
机甲类型的电影通常包含大量机器人或者机械元素。因此，\'''类型为机甲\'''可以用\'''有机器人或机械元素的\'''来替代。
0-2岁的年龄范围意味着影片需要适合幼儿观看，所以\'''年龄范围为0-2岁\'''可以用\'''适合幼儿观看\'''来替代。
最后，\'''付费类型为免费\'''说明电影无需付费观看，可用\'''免费观看\'''来替代。
user:我想看一些有机器人或者机械元素的电影，适合幼儿观看的，而且不用付费的。

##
目的:'''
    re = generate_demand(message['response'], tree)
    user_msg = prompt + purpose + "\n客服问题:您好，请问您有什么需求呢？" + "\n需求列表:\n" + re
    encoded_user_msg = quote(user_msg)

    resp = requests.post(f"http://127.0.0.1:8081/process_messages?question={encoded_user_msg}")
    if resp.status_code == 200:
        return resp.text


class DEMAND:
    def __init__(self):
        self.input = ""
        self.response = {}


from tqdm import tqdm

re_contexts = []
for key, context_data in tqdm(contexts.items()):
    page_source = key
    ami_tree = trees[page_source]
    purpose = purposes[page_source]
    re_contexts = []
    tree_json = []
    for context in tqdm(context_data):
        random_number = random.randint(0, 10)
        demand = DEMAND()
        if random_number < 8:
            re_context = gpt(context, ami_tree, purpose)
            if re_context is not None:
                if "user:" in re_context:
                    start_index = re_context.find("user:") + len("user:")
                    extracted_statement = re_context[start_index:].strip()
                else:
                    extracted_statement = re_context
            else:
                extracted_statement = re_context
            demand.input = extracted_statement
        else:
            demand.input = context['input']
        demand.response = context["response"]
        re_contexts.append(demand)
    for con in re_contexts:
        tree_dict = {
            "input": con.input,
            "response": con.response
        }
        tree_json.append(tree_dict)
    tree_dict = {
        "page_source": page_source,
        "content": tree_json
    }
    with open("re_write.json", "a", encoding="utf-8") as f:
        json.dump(tree_dict, f, indent=4, ensure_ascii=False)
