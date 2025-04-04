import copy
import random
from default_data import WEBPAGE


# 按照策略选择要询问的token
def select_token(qtoken_list, processing_list):
    """选择下一个要询问的token

    :param qtoken_list: 剩余的token列表
    :param processing_list: 正在处理的token列表
    :return: 选择的token
    """
    selectable_tokens = []
    for token in qtoken_list:
        if token not in processing_list or processing_list[token].infer_value is None: # Check if NOT in processing OR infer_value is None
            selectable_tokens.append(token)

    if not selectable_tokens:
        return None
    # 随机选择一个 token
    q_token = random.choice(selectable_tokens)
    return q_token


# 保存符合需求的key和对应value
def save_qualify_value(kvs, selected_qtoken, webs, feature):
    token_type = selected_qtoken.token_type
    if token_type == "选项":
        for page in selected_qtoken.pages:  # 对于含有这个token的page
            index = page["page_number"] - 1
            keys = page["value"]
            for key in keys:  # 对于某个网页中符合token的一个key
                for leaf in webs[index].tree:  # 对于该网页的kv树
                    if leaf["key"] == key:
                        if feature.value_type != "范围" and feature.infer_value in leaf["value"]:
                            # 用户的需求value在对应key的value中
                            # 将用户需求记录，这里不是token了，已经变成key了
                            new_tuple = (key, feature.infer_value)
                            if index + 1 in list(kvs.keys()):
                                kvs[index + 1].append(new_tuple)
                            else:
                                kvs[index + 1] = [new_tuple]
                            # demand是map
                        elif feature.value_type == "范围":  # 添加这部分处理范围
                            for value in leaf["value"]:
                                if value in feature.options and feature.options[value] == feature.infer_value :
                                    new_tuple = (key, value) #改为value
                                    if index + 1 in list(kvs.keys()):
                                        kvs[index + 1].append(new_tuple)
                                    else:
                                        kvs[index + 1] = [new_tuple]
                                    break


# 从possible_pages中移除不含用户要求的page，保持序号
def remove_page(possible_pages, qtoken):
    qualify_page_numbers = [page["page_number"] for page in qtoken.pages]
    # 重构
    for i in range(0, len(possible_pages)):
        if i + 1 not in qualify_page_numbers:
            possible_pages[i] = WEBPAGE()


# 从remain_demand中移除特定的demand
def remove_from_demand(remain_demand, param):
    if param in remain_demand.token:
        demand_index = remain_demand.token.index(param)
        remain_demand.token.remove(param)
        remain_demand.value.remove(remain_demand.value[demand_index])
        remain_demand.key.remove(remain_demand.key[demand_index])
