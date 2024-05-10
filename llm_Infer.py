from functools import reduce

from gpt_api import formulate_gpt_question
import requests
from urllib.parse import quote
import json


# infer函数一次只涉及一个token
def feature_value_infer_llm(qtoken_list, webs, token, sentence):
    # 处理token，将所有具有该token的页面的value求并集
    all_values = []
    # qtoken中存储了页面信息
    qtoken = qtoken_list[token]
    for page in qtoken.pages:
        index = page["page_number"] - 1
        keys = page["value"]
        for key in keys:  # 对于某个网页中符合token的一个key
            for leaf in webs[index].tree:  # 对于该网页的kv树
                if leaf["key"] == key:
                    all_values.append(leaf["value"])
    # 该token对应的所有value
    union_set = list(reduce(set.union, map(set, all_values), set()))
    # 过滤无效value
    incompetent = ["不限", "自定义", "其他", "更多", "全部"]
    filtered_set = [element for element in union_set if not any(incomp in element for incomp in incompetent)]
    if qtoken.token_type == "数值":
        results, _ = formulate_gpt_question(None, filtered_set, None, "sort")  # 将数值型的value排序
        start_index = results.find("['")
        end_index = results.find("']")
        filtered_set = results[start_index + 2:end_index].split("', '")
    # 构造语句
    user_msg, history = "### 指令:\n下述提供了user demand，target feature以及options。user demand为user对于一个或多个feature的描述。根据user " \
                        "demand推测并在options中选择一个最适合target feature的值。若推测的值不明确，则结果为None。\n\n### 输入:\nuser " \
                        "demand:\'''" + sentence + "。\'''\ntarget feature:#" + token + "#\noptions:\n'''" + \
                        str(filtered_set) + "'''", []
    encoded_user_msg = quote(user_msg)
    invalid = True
    infer_value = None
    while invalid is True:
        resp = requests.post(f"http://127.0.0.1:8081/predict?user_msg={encoded_user_msg}", json=history)
        if resp.status_code == 200:
            infer_token = resp.json()['key']
            infer_value = resp.json()['value']
            if infer_value == 'None':
                infer_value = None
            if infer_token != token or infer_value not in filtered_set:
                invalid = True
            else:
                invalid = False

    # 网络错误或者无用户所需的值
    return infer_value, filtered_set


# infer函数一次只涉及一个token
def feature_value_infer(qtoken_list, webs, token, sentence):
    # 处理token，将所有具有该token的页面的value求并集
    all_values = []
    # qtoken中存储了页面信息
    qtoken = qtoken_list[token]
    for page in qtoken.pages:
        index = page["page_number"] - 1
        keys = page["value"]
        for key in keys:  # 对于某个网页中符合token的一个key
            for leaf in webs[index].tree:  # 对于该网页的kv树
                if leaf["key"] == key:
                    all_values.append(leaf["value"])
    # 该token对应的所有value
    union_set = list(reduce(set.union, map(set, all_values), set()))
    # 过滤无效value
    incompetent = ["不限", "自定义", "其他", "更多", "全部"]
    filtered_set = [element for element in union_set if not any(incomp in element for incomp in incompetent)]
    if qtoken.token_type == "数值":
        results, _ = formulate_gpt_question(None, filtered_set, None, "sort")  # 将数值型的value排序
        start_index = results.find("['")
        end_index = results.find("']")
        filtered_set = results[start_index + 2:end_index].split("', '")
    # 构造语句
    user_msg, history = "输入的指令是:''' " + sentence + " '''\n 请抽取#" + token + \
                        "#\n如果info的提供了候选值,那就请从中选取, info如下\n'''" + \
                        str(filtered_set) + "'''", []
    print(user_msg)
    infer_value = input("input infer_value:")
    if infer_value == "None":
        infer_value = None
    # 网络错误或者无用户所需的值
    return infer_value, filtered_set


def generate_examples(options):
    result = ",".join(options)
    return result


def token_infer(answer, token_list, features_prompt):
    token_results = []
    prompt = ""
    for feature, pair in features_prompt.items():
        annotation, example_value = pair
        examples = generate_examples(example_value)
        prompt = prompt + feature + ":" + annotation + ";如" + examples + " 等\n"
    infer, _ = formulate_gpt_question(None, answer, token_list, "token_infer", prompt)
    invalid = ["关键词", ":", "："]
    for k in invalid:
        if k in infer:
            infer = infer.replace(k, "")
    add_tokens = []
    if "，" in infer:
        add_tokens = infer.split("，")
    elif "," in infer:
        add_tokens = infer.split(",")
    else:
        add_tokens.append(infer)

    if len(token_list) > 0:
        for token in token_list:
            add_tokens.append(token)
        if 'None' in add_tokens:
            add_tokens.remove('None')
    # 去重
    add_tokens = list(set(add_tokens))

    token_results.extend(add_tokens)
    return token_results


def planning_user_profiles():
    pass
