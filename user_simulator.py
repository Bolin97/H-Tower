import json
import random

import default_data
import webpage
from cal_average_length import cal
from default_data import DEMAND
from gpt_api import formulate_gpt_question


# webPages = webpage.webPages

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
def expression_generation(class_type, question, ktoken, kvalue, remain, qtoken_list, user_profiles):
    tokens = [ktoken]
    values = [kvalue]
    is_add = random.randint(1, 10)
    if 8 < is_add:  # %20的概率增加别的value
        feature_num = random.randint(2, 3)  # 增加1-2个feature
        if len(remain.token) > feature_num:
            while len(tokens) < feature_num:
                num = random.randint(0, len(remain.token) - 1)  # 额外的value数量
                # 用户在回答语句中额外携带的value（以及对应token）
                token = remain.token[num]
                # 检查 token 是否已经在 param_1 中
                if token not in tokens:
                    tokens.append(token)
                    values.append(remain.value[num])
    answer_type = None
    answer = ""
    while answer_type != default_data.EXPLICIT :
        answer, answer_type ,_,_= formulate_gpt_question(class_type, question, qtoken_list, "answer",
                                                         tokens, values)
    annotations=[]
    for a in tokens:
        if a in qtoken_list.keys():
            annotations.append(qtoken_list[a].annotation)
    if "\n回答" in answer:
        origin_answer = answer
        start_index = origin_answer.find("\n回答") + len("\n回答") + 1
        extracted_statement = origin_answer[start_index:].strip()
        answer = extracted_statement
    if answer_type == default_data.RE_VAGUE:
        # 用user profile重写
        result = ""
        for token in tokens:
            if token in user_profiles:
                result = result + str(token) + "(" + str(qtoken_list[token].annotation) + "):\n"
        if result != "":
            re_answer, _ ,_,_= formulate_gpt_question(class_type, result, answer, "add_user_profile",
                                                      user_profiles, tokens)
            answer = re_answer
    return answer, answer_type, tokens, values,annotations


def retell_generator(class_type, process_info, token, value, qtoken_list, user_profiles):
    answer_type = None
    answer = ""
    question = "请问你对" + token + "有什么偏好吗"
    tokens = [token]
    values = [value]
    while answer_type != default_data.EXPLICIT :
        answer, answer_type ,_,_= formulate_gpt_question(class_type, question, qtoken_list, "answer",
                                                         tokens, values)

    if "\n回答" in answer:
        origin_answer = answer
        start_index = origin_answer.find("\n回答") + len("\n回答") + 1
        extracted_statement = origin_answer[start_index:].strip()
        answer = extracted_statement
    if answer_type == default_data.RE_VAGUE:
        # 用user profile重写
        result = ""

        if token in user_profiles:
            result = result + str(token) + "(" + str(qtoken_list[token].annotation) + "):\n"
        if result != "":
            tokens = [token]
            re_answer, _ ,_,_= formulate_gpt_question(class_type, result, answer, "add_user_profile",
                                                      user_profiles, tokens)
            answer = re_answer
        history = []
        for dialog in process_info.dialog_history:
            if "user" in dialog:
                history.append(dialog)
        retell_answer, _ ,_,_= formulate_gpt_question(class_type, token, qtoken_list[token].annotation, "user_retell",
                                                      history, answer)
    return answer, answer_type


# 开放式回答
def user_choose_token(class_type, question, remain_demand, qtoken_list, user_profiles):
    kv = {}
    if len(remain_demand.token) >= 1:
        answer, answer_type, keys, values = trigger_generator(class_type, qtoken_list, "open_ended_answer",
                                                              remain_demand, user_profiles, question)

        for token, value in zip(keys, values):
            kv[token] = value
    else:
        keys = []
        values = []
        answer, answer_type ,_,_= formulate_gpt_question(class_type, question, None, "end")
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
        if token not in keys:
            keys.append(token)
            values.append(value)
            if token not in qtoken_list:
                continue
            annotations.append(qtoken_list[token].annotation)
            token_types.append(qtoken_list[token].token_type)

    if s_type == "trigger":
        trigger, gpt_type ,_,_= formulate_gpt_question(class_type, keys, values, "trigger", annotations)
        return trigger, gpt_type, keys, values
    else:
        # 开放式问答
        answer_type = None
        answer = ""
        while answer_type != default_data.EXPLICIT :
            answer, answer_type ,_,_= formulate_gpt_question(class_type, ques, qtoken_list, "answer",
                                                             keys, values)
        answer = answer
        if answer_type == default_data.RE_VAGUE:
            # 用user profile重写
            result = ""
            for token in keys:
                if token in user_profiles:
                    result = result + str(token) + "(" + str(qtoken_list[token].annotation) + "):\n"
            if result != "":
                re_answer, _ ,_,_= formulate_gpt_question(class_type, result, answer, "add_user_profile",
                                                          user_profiles, keys)
                answer = re_answer
        return answer, answer_type, keys, values


# 需求判断器
def requirement_judgment(class_type, user_value, infer_value, sentence):
    answer, answer_type ,_,_= formulate_gpt_question(class_type, sentence, user_value, "judge", infer_value)
    result, _ ,_,_= formulate_gpt_question(class_type, answer, None, None)
    if "赞同" in result or user_value == infer_value:
        result = True
    elif "反对" in result:
        result = False
    return result, answer, answer_type


# infer token判断
def infer_token_judgement(question, infer_token, user_tokens):
    if infer_token in user_tokens:
        answer, _ ,_,_= formulate_gpt_question(None, question, None, "agree")
    else:
        answer, _ ,_,_= formulate_gpt_question(None, question, None, "disagree")
    if "不" in answer or "没" in answer:
        emotion = "消极"
    else:
        emotion = "积极"
        # emotion, _ ,_,_= formulate_gpt_question(None, question, answer, 'emotion')
    return emotion, answer


# verify list检查
def check_verify_list(user_said_demand, verify_list, processing_list):
    no_need_token = []
    wrong_token_value = []
    none_token_value = []
    missed_token_value = []
    for token in verify_list.keys():
        if token not in processing_list.token:
            no_need_token.append(token)
        else:
            if verify_list[token] not in processing_list.value:
                wrong_token_value.append(token)

    # 注意操作对象，processing_list是agent使用的对象，所以要对user_said_demand进行操作
    for token in user_said_demand.token:
        if token not in verify_list.keys():
            missed_token_value.append(token)
    return no_need_token, wrong_token_value, none_token_value, missed_token_value


# 需求验证器
def requirement_validation(class_type, validate_type, token_list, qtoken_list):
    if validate_type == 'no_need':
        answer, answer_type ,_,_= formulate_gpt_question(class_type, token_list, qtoken_list, "no_need")
    elif validate_type == 'wrong_value':
        answer, answer_type ,_,_= formulate_gpt_question(class_type, token_list, qtoken_list, "wrong_value")
    else:
        answer, answer_type ,_,_= formulate_gpt_question(class_type, token_list, qtoken_list, "missed_token")
    return answer


# 需求辅助选择器
def requirement_confirmation(class_type, question, value, que_type, middle_value=None, sorted_map=None):
    answer = None
    answer_type = None
    if que_type == "options_exist":
        answer, answer_type ,_,_= formulate_gpt_question(class_type, question, value, "select", "selection")
    elif que_type == "options_not_exist":
        answer, answer_type ,_,_= formulate_gpt_question(class_type, question, value, "select", "not_exist")
    elif que_type == "numeric":
        result, _ ,_,_= formulate_gpt_question(class_type, middle_value, value, "numeric_result")
        if "A" in result:
            answer, answer_type ,_,_= formulate_gpt_question(class_type, question, value, "lower_number", que_type)
        elif "C" in result:
            answer, answer_type ,_,_= formulate_gpt_question(class_type, question, value, "higher_number", que_type)
        elif "B" in result:
            answer, answer_type ,_,_= formulate_gpt_question(class_type, question, value, "right_number", que_type)
    elif que_type == "range":
        agent_value = sorted_map[middle_value]
        user_value = sorted_map[value]
        if agent_value[1] <= user_value[0]:
            answer, answer_type ,_,_= formulate_gpt_question(class_type, question, value, "higher_number", que_type)
        elif agent_value[0] >= user_value[1]:
            answer, answer_type ,_,_= formulate_gpt_question(class_type, question, value, "lower_number", que_type)
        elif agent_value[0] == user_value[0] and agent_value[1] == user_value[1]:
            answer, answer_type ,_,_= formulate_gpt_question(class_type, question, value, "right_number", que_type)
        else:
            answer, answer_type ,_,_= formulate_gpt_question(class_type, question, value, "approximate_number", que_type)

    return answer, answer_type
