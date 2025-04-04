import json
from collections import defaultdict
from default_data import PAGEKEY


def cal(webPages, domain_path, class_type=None):
    TF = {}
    page_numbers = []
    # 如果是制定子领域的
    if class_type:
        path_0 = "data/" + domain_path + "/" + domain_path + ".json"
        with open(path_0, "r", encoding="utf-8") as file:
            domain_data = json.load(file)
        for d in domain_data:
            if d["type"] == class_type:
                page_numbers.append(d["page_index"])
    # 计算每一个token的item长度（平均值）
    path = "data/" + domain_path + "/" + domain_path + "_token_tf.json"
    with open(path, "r", encoding="utf-8") as file:
        tf = json.load(file)
    page_key2token = {}
    page_keys = []
    for data in tf:
        token = data["token"]
        token_TF = 0
        for page in data["pages"]:
            if class_type is None or page["page_number"] in page_numbers:
                pageKey = PAGEKEY()
                token_TF = token_TF + 1
                # pageKey{token,pageNum,key}
                pageKey.token = token
                pageKey.page_num = page["page_number"]
                pageKey.key = page["value"]
                # key和web index一起索引对应的token
                for k in pageKey.key:
                    key = (pageKey.page_num, k)
                    page_key2token[key] = token
                page_keys.append(pageKey)
        if token_TF != 0:
            TF[token] = token_TF

    for pageKey in page_keys:
        index = pageKey.page_num - 1
        length = 0
        for key in pageKey.key:  # 对于某个网页中符合token的一个key
            for leaf in webPages[index].tree:
                if leaf["key"] == key:
                    length += len(leaf["value"])
        pageKey.amount = length / len(pageKey.key)  # 求平均

    token_dict = defaultdict(list)
    for pageKey in page_keys:
        token_dict[pageKey.token].append(pageKey)

    # 计算每个 token 对应元素的 amount 的平均值
    token_averages = {}
    for token, token_pagekeys in token_dict.items():
        total_amount = sum(pagekey.amount for pagekey in token_pagekeys)
        average_amount = total_amount / len(token_pagekeys) if len(token_pagekeys) > 0 else 0
        token_averages[token] = average_amount
    valid_tokens = set()
    for pk in page_keys:
        valid_tokens.add(pk.token)
    # print("每个token对应的amount的平均数:")
    # print(token_averages)
    return token_averages, page_key2token, valid_tokens, TF, page_numbers
