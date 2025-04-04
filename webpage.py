import json
from default_data import WEBPAGE


# 网页的相关信息
def load_webpages(domain_path, domain, class_type=None):
    path_1 = "data/" + domain_path + "/" + domain_path + ".json"
    with open(path_1, "r", encoding='utf-8') as file:
        data = json.load(file)

    path_2 = "data/" + domain_path + "/" + domain_path + "_tree.json"
    with open(path_2, "r", encoding='utf-8') as file2:
        data2 = json.load(file2)
        tree_dict = {tree["page_source"]: tree["content"] for tree in data2}
    webPages = [
        WEBPAGE(web["page_index"], web["image"], web["website"], domain, web["type"], web["required_tokens"],
                tree_dict.get(web["image"], []))
        for index, web in enumerate(data)
        if not class_type or web["type"] == class_type
    ]

    for webPage in webPages:
        for item in webPage.tree:
            webPage.keys.append(item["key"])
    return webPages
