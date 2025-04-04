import copy
import re
from functools import reduce
from urllib.parse import quote
from openai import OpenAI
import requests

from gpt_api import formulate_gpt_question,chat


"""# infer函数一次只涉及一个token
def feature_value_infer_llm(qtoken_list, webs, token, sentence, user_profile=None):
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
    # if qtoken.token_type == "数值":
    #     results, _ = formulate_gpt_question(None, filtered_set, None, "sort")  # 将数值型的value排序
    #     start_index = results.find("['")
    #     end_index = results.find("']")
    #     filtered_set = results[start_index + 2:end_index].split("', '")
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
"""

# infer函数一次只涉及一个token
def feature_value_infer_llm(qtoken_list, webs, token, sentence, user_profile=None):
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

    # 构造GPT prompt
    instruction = "你是一个助手，你需要根据用户的需求和给定的选项，推断出用户最想要的选项。\n" \
                  "请从【选项】列表中选择**一个且仅一个**最符合用户需求的选项，并**只返回选项的文本内容，不要包含任何其他文字、解释或标点符号。**\n" \
                  "如果无法明确推断，或者选项中没有合适的，请回答 'None'。"

    user_demand_prompt = f"用户需求: '{sentence}'"
    target_feature_prompt = f"目标特征: {token}"
    options_prompt = f"选项: {str(filtered_set)}"

    question = f"{instruction}\n\n{user_demand_prompt}\n{target_feature_prompt}\n{options_prompt}"

    gpt_response,_ = chat(question)
    infer_value = None

    if gpt_response:
        gpt_response = gpt_response.strip() # 去除首尾空格
        if gpt_response == "None":
            infer_value = None
        elif gpt_response in filtered_set:
            infer_value = gpt_response
        else:
            # 如果GPT返回的值不在选项中，也设为None
            print(f"GPT返回的值 '{gpt_response}' 不在选项列表中。")
            infer_value = None

    return infer_value, filtered_set

# infer函数一次只涉及一个token
def feature_value_infer(qtoken_list, webs, token, sentence, user_profile=None):
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
    possible_values = str(filtered_set)
    if qtoken.token_type != "选项":
        new_set = copy.deepcopy(filtered_set)
        filtered_set = sorted_range(new_set)
        possible_values = str(list(filtered_set.keys()))
    # 构造语句
    print("\n[以下的过程是llm抽取slot的过程模拟]")
    additional = ""

    if user_profile is not None:
        if token in user_profile:
            profile = user_profile[token]
            additional = "已知" + profile.profile_key + "为" + profile.profile_value + "，并且，" + profile.profile_info
    user_msg, history = "### 指令:\n下述提供了user demand，target feature以及options。" + \
                        "user demand为user对于一个或多个feature的描述。根据user demand推测并在options中选择一个最适合target " \
                        "feature的值。若推测的值不明确，则结果为None。" + \
                        "### 输入:user demand:''' " + sentence + additional + " '''\ntarget feature:#" + token + \
                        "#\noptions:\n'''" + \
                        possible_values + "'''", []
    print(user_msg)
    infer_value = input("input infer_value:")
    print("[以上的过程是llm抽取slot的过程模拟]\n")
    if infer_value == "None":
        infer_value = None
    # 网络错误或者无用户所需的值
    return infer_value, filtered_set


def sorted_range(options):
    # 定义正则表达式模式
    pattern = r'\d+'
    options_map = {}
    # 提取数字
    for text in options:
        numbers = re.findall(pattern, text)
        if len(numbers) == 1:
            if "上" in text or ">" in text or "大" in text:
                options_map[text] = [int(numbers[0]), float('inf')]
            else:
                options_map[text] = [float('-inf'), int(numbers[0])]
        else:
            options_map[text] = [int(numbers[0]), int(numbers[1])]

    # 排序
    sorted_map = dict(sorted(options_map.items(), key=lambda item: (item[1][0], item[1][1])))
    return sorted_map


def token_infer(answer, token_list, features_prompt):
    prompt = ""
    for feature, pair in features_prompt.items():
        annotation, example_value = pair
        examples = ",".join(example_value)
        prompt = prompt + feature + ":" + annotation + ";如" + examples + " 等\n"
    infer, _ ,_,_= formulate_gpt_question(None, answer, None, "token_infer", prompt)
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
    # 去重
    add_tokens = list(set(add_tokens))
    infer_tokens = []
    if token_list is None: # None check added here
        token_list= []
    if len(add_tokens) > 1:
        infer_tokens = [token for token in add_tokens if token in token_list]
    return infer_tokens


def planning_user_profiles():
    pass
