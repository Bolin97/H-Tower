import json
import random
import time

from default_data import DEMAND_LIST, USERPROFILE, DEMAND
from gpt_api import formulate_gpt_question
from user_simulator import trigger_generator, construct_requirements
from tqdm import tqdm

# 选择领域和子类
# 构建需求列表
demand_datas = []
demand_qtoken_list = None
domain = None
class_type = None
user_demand = DEMAND()


def find_suggest_profile_value(text):
    end_index = text.rfind(']')
    if end_index != -1:
        # 从找到的']'往前找到最近的'['
        start_index = text.rfind('[', 0, end_index)
        if start_index != -1:
            # 提取中括号中的信息并去除单引号
            result = text[start_index + 1:end_index]
            if "'" in result:
                result = result.replace("'", "")
            return result
    return None


def trigger_and_profile(index, user_profiles, token_index):
    global demand_qtoken_list, domain, class_type, demand_datas, user_demand
    trigger_sen, trigger_type, user_tokens, user_values = trigger_generator(class_type, demand_qtoken_list, "trigger",
                                                                            user_demand)

    data = DEMAND_LIST()
    data.id = index

    data.user_profile = {}
    if trigger_type != "不含需求":
        data.trigger_features = user_tokens
        start_index = trigger_sen.find("开场白") + len("开场白") + 1
        extracted_statement = trigger_sen[start_index:].strip()
        data.trigger = extracted_statement
    else:
        data.trigger_features = []
        data.trigger = trigger_sen
    data.trigger_type = trigger_type
    data.domain = domain
    data.features = user_demand.token
    data.demand_type = class_type
    data.page_number = user_demand.page_number
    need = {}
    for demand_index in range(0, len(user_demand.key)):
        need[user_demand.key[demand_index]] = user_demand.value[demand_index]
    data.need = need
    # 生成user_profile
    for feature_index, feature in enumerate(user_demand.token):
        value = user_demand.value[feature_index]
        if feature in token_index.keys():
            profile_index = token_index[feature]
            profile = user_profiles[profile_index]
            random_info = random.choice(list(profile.user_info.keys()))
            random_number = random.randint(0, 10)
            if random_number < 8:  # 80%的概率有用户profile
                # user_info的值和value一样(例如区域和公司所在地)
                if profile.user_info[random_info] == "None":
                    data.user_profile[random_info] = value
                # user_info的值是由value推理来的
                else:
                    pair = (feature, value)
                    result, _ = formulate_gpt_question(class_type, pair, demand_qtoken_list, "user_profile", profile,
                                                       random_info)
                    result = find_suggest_profile_value(result)
                    data.user_profile[random_info] = result

    demand_datas.append(data)


def run():
    global user_demand, demand_qtoken_list, domain, class_type, demand_datas

    user_demand, demand_qtoken_list, class_type, path, domain = construct_requirements()

    with open("./data/" + path + "/user_profile.json", "r", encoding="utf-8") as file:
        profiles = json.load(file)
    user_profiles = [
        USERPROFILE(profile["token"], profile["user_info"], profile["information"])
        for profile in profiles]
    token_index = {}
    for index, profile in enumerate(user_profiles):
        token_index[profile.token] = index
    trigger_and_profile(i, user_profiles, token_index)


if __name__ == '__main__':
    start_time = time.time()
    # 防止网络原因导致进程终止，所以每次循环次数设置为10
    for i in tqdm(range(952, 1001)):
        run()
        if (i % 50 == 1 and i != 1) or i == 1000:
            demands_json = []
            for dem in demand_datas:
                conversation_dict = {
                    "id": dem.id,
                    "user_profile": dem.user_profile,
                    "trigger": dem.trigger,
                    "trigger_features": dem.trigger_features,
                    "trigger_type": dem.trigger_type,
                    "domain": dem.domain,
                    "demand_type": dem.demand_type,
                    "features": dem.features,
                    "qualify_page": dem.page_number,
                    "need": dem.need
                }
                demands_json.append(conversation_dict)

            # 此处是add
            with open("data/demands.json", "a", encoding="utf-8") as f:
                json.dump(demands_json, f, indent=4, ensure_ascii=False)
            demand_datas = []

    end_time = time.time()
    run_time = end_time - start_time
    print(run_time)
