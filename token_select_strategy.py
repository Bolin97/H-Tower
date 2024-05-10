from default_data import WEBPAGE


# 按照策略选择要询问的token
def select_token(qtoken_list):
    min_tf = float('-inf')  # 更改初始值为负无穷，以便找到最大的 TF
    min_average_item_num = float('inf')
    selected_token = None
    for token, qtoken in qtoken_list.items():
        if qtoken.TF > min_tf or (qtoken.TF == min_tf and qtoken.average_item_num < min_average_item_num):
            min_tf = qtoken.TF
            min_average_item_num = qtoken.average_item_num
            selected_token = token
    return selected_token


# 保存符合需求的key和对应value
def save_qualify_value(kvs, selected_qtoken, webs, kvalue):
    token_type = selected_qtoken.token_type
    if token_type == "选项":
        for page in selected_qtoken.pages:  # 对于含有这个token的page
            index = page["page_number"] - 1
            keys = page["value"]
            for key in keys:  # 对于某个网页中符合token的一个key
                for leaf in webs[index].tree:  # 对于该网页的kv树
                    if leaf["key"] == key and kvalue in leaf["value"]:  # 用户的需求value在对应key的value中
                        # 将用户需求记录，这里不是token了，已经变成key了
                        new_tuple = (key, kvalue)
                        # demand是map
                        if index + 1 in list(kvs.keys()) :
                            kvs[index + 1].append(new_tuple)
                        else:
                            kvs[index + 1] = [new_tuple]


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
