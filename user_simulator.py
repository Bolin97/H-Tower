import json
import random

import default_data
import webpage
from cal_average_length import cal
from default_data import DEMAND
from gpt_api import formulate_gpt_question

# webPages = webpage.webPages
add = default_data.is_add


# 需求生成
def construct_requirements():
    # 选择领域和子类
    with open("data/path_to_web.json", "r", encoding='utf-8') as file:
        data = json.load(file)
    random_demand = random.choice(data)
    # 随机选择某领域的某个子类
    domain = random_demand["domain"]
    domain_path = random_demand["path"]
    types = random_demand["type"]
    class_type = random.choice(types)

    print("Demand:", domain)
    print("Path:", domain_path)
    print("Type: ", class_type)
    # 领域内的web
    webPages = webpage.load_webpages(domain_path, domain)
    token_averages, page_key2token, _, _, _ = cal(webPages, domain_path)
    # page_key2token key转为对应token
    qtoken_list = default_data.read_token(token_averages, domain_path)

    while True:
        user_demand = DEMAND()
        page_index = random.randint(0, len(webPages) - 1)  # 随机选择一个页面然后再选key，防止最后没有页面符合要求
        if webPages[page_index].class_type == class_type:
            user_demand.page_number = page_index + 1  # 网页编号,真实编号，从1开始
            required_tokens = webPages[page_index].required_tokens  # 必填token
            key_max = min(len(webPages[page_index].keys) - 1, 10) - len(required_tokens)
            if 4 <= key_max:
                break

    key_num = random.randint(4, key_max)  # 随机选择key的数量,不超过10个
    selected_trees = []
    selected_tokens = set()
    while len(selected_trees) < key_num:
        # 从webPages[page_index].tree中随机选择一个tree
        selected_tree = random.choice(webPages[page_index].tree)
        token = page_key2token.get((user_demand.page_number, selected_tree["key"]))
        # 如果已经选择的tree中没有相同的token，则将其加入选择列表
        if token not in selected_tokens:
            if ('单价' in selected_tokens and '总价' in selected_tokens) or (
                    '总价' in selected_tokens and '单价' in selected_tokens):
                continue
            selected_trees.append(selected_tree)
            selected_tokens.add(token)

    # 往需求列表中加入必选token
    for re_token in required_tokens:
        if re_token not in selected_tokens:
            for tree in webPages[page_index].tree:
                key_to_token = page_key2token.get((user_demand.page_number, tree["key"]))
                if key_to_token == re_token:
                    selected_trees.append(tree)
                    selected_tokens.add(re_token)
                    break

    # 过滤无效value
    for kv in selected_trees:
        user_demand.key.append(kv["key"])
        incompetent = ["不限", "自定义", "其他", "更多", "全部"]
        selected_value = random.sample(kv["value"], 1)[0]
        while any(incomp in selected_value for incomp in incompetent):
            selected_value = random.sample(kv["value"], 1)[0]
        user_demand.value.append(selected_value)

    print("\n需求列表：")
    for i in range(0, len(user_demand.key)):
        user_demand.token.append(page_key2token.get((user_demand.page_number, user_demand.key[i])))
        print(user_demand.key[i] + ":" + user_demand.value[i])
    print("\n")
    return user_demand, qtoken_list, class_type, domain_path, domain


# 表达生成
def expression_generation(question, ktoken, kvalue, remain, ex_type, qtoken_list, user_profiles):
    tokens = []
    values = []
    token_types = []
    annotations = []
    tokens.append(ktoken)
    values.append(kvalue)
    if ex_type == "question":
        global add
        if 8 < add:  # %20的概率增加别的value
            feature_num = random.randint(0, 2)
            while len(tokens) < feature_num:
                num = random.randint(0, len(remain.token) - 1)  # 额外的value数量
                # 用户在回答语句中额外携带的value（以及对应token）
                token = remain.token[num]
                value = remain.value[num]

                # 检查 token 是否已经在 param_1 中
                if token not in tokens:
                    tokens.append(token)
                    values.append(value)
        for token in tokens:
            annotations.append(qtoken_list[token].annotation)
            token_types.append(qtoken_list[token].token_type)
        answer, answer_type = formulate_gpt_question(user_profiles, question, values, "answer",
                                                     tokens, annotations, token_types)
        return answer, answer_type, tokens, values
    elif ex_type == "retell":
        for token in tokens:
            annotations.append(qtoken_list[token].annotation)
        answer, answer_type = formulate_gpt_question(user_profiles, question, values, "answer",
                                                     tokens, annotations, token_types)
        return answer, answer_type, tokens, values


# 开放式回答
def user_choose_token(class_type, question, remain_demand, qtoken_list, user_profiles):
    kv = {}
    if len(remain_demand.token) >= 1:
        answer, answer_type, keys, values = trigger_generator(class_type, qtoken_list, "open_ended_answer",
                                                              remain_demand, user_profiles, question)

        for index, token in enumerate(keys):
            kv[token] = values[index]
    else:
        keys = None
        values = None
        answer, answer_type = formulate_gpt_question(class_type, question, None, "end")
    return answer, answer_type, keys, values, kv


# 触发句
def trigger_generator(class_type, qtoken_list, s_type, remain_demand, user_profiles=None, ques=None):
    # 随机选择1-2个特征生成user语句
    remain_num = len(remain_demand.token)
    max_num = min(remain_num, 2)
    feature_num = random.randint(1, max_num)
    keys = []
    annotations = []
    values = []
    token_types = []
    while len(keys) < feature_num:
        # key 和 value 对应
        # 随机选择token
        num = random.randint(0, len(remain_demand.token) - 1)
        token = remain_demand.token[num]
        value = remain_demand.value[num]

        # 检查 token 是否已经在 param 中，防止重复选择
        if token not in keys and token in list(qtoken_list.keys()):
            keys.append(token)
            values.append(value)
            annotations.append(qtoken_list[token].annotation)
            token_types.append(qtoken_list[token].token_type)

    if s_type == "trigger":
        trigger, gpt_type = formulate_gpt_question(class_type, keys, values, "trigger", annotations)
        return trigger, gpt_type, keys, values
    else:
        # 开放式问答
        answer, answer_type = formulate_gpt_question(user_profiles, ques, values, "answer", keys, annotations,
                                                     token_types)
        return answer, answer_type, keys, values


# 需求判断器
def requirement_judgment(class_type, user_value, infer_value, sentence):
    answer, answer_type = formulate_gpt_question(class_type, sentence, user_value, "judge", infer_value)
    result, _ = formulate_gpt_question(class_type, answer, None, None)
    if "赞同" in result:
        result = True
    elif "反对" in result:
        result = False
    return result, answer, answer_type


def check_verify_list(user_said_demand, verify_list, processing_list):
    no_need_token = []
    wrong_token_value = []
    none_token_value = []
    missed_token_value = []
    for token in verify_list:
        if token not in user_said_demand.token:
            # 前几次回答中没有的【排除了错误的token】
            no_need_token.append(token)
        else:
            true_value = user_said_demand.value[user_said_demand.token.index(token)]
            if processing_list[token].infer_value != true_value:
                wrong_token_value.append(token)
    # 注意操作对象，processing_list是agent使用的对象，所以要对user_said_demand进行操作
    for token in user_said_demand.token:
        if token not in list(processing_list.keys()):
            missed_token_value.append(token)
        elif processing_list[token].infer_value is None:
            none_token_value.append(token)
    return no_need_token, wrong_token_value, none_token_value, missed_token_value


# 需求验证器
def requirement_validation(class_type, validate_type, token_list, processing_list=None):
    if validate_type == 'no_need':
        pass
    elif validate_type == 'wrong_value':
        pass
    # if value is None:
    #     while "说明具体" in answer:
    #         answer, _ = formulate_gpt_question(class_type, token, None, "missing_token_value")
    # else:
    #     while "说明具体" in answer:
    #         answer, _ = formulate_gpt_question(class_type, token, value, "wrong_token")
    # print("user:", answer)
    # return answer


# 需求辅助选择器
def requirement_confirmation(class_type, question, value, que_type):
    if que_type == "options":
        if value in question:
            answer, answer_type = formulate_gpt_question(class_type, question, value, "select", "selection")
        else:
            answer, answer_type = formulate_gpt_question(class_type, question, value, "select", "not_exist")
        return answer, answer_type
    # elif que_type ==  "numeric":
