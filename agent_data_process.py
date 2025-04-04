import re


# 对可能的领域打分
def spilt_score(data, types):
    """

    :param data: agent的回答
    :param types: 待选的types
    :return:
    """
    keywords_list = types.split(',')
    pattern = r'(' + '|'.join(map(re.escape, keywords_list)) + r'):(\d+)'
    matches = re.findall(pattern, data)
    scores_dict = {key: int(value) for key, value in matches}
    scores = dict(sorted(scores_dict.items(), key=lambda item: item[1], reverse=True))
    keys_with_score_over_5 = []
    sorted(scores.values(), reverse=True)
    for key, value in scores.items():
        if value >= 9:
            return [key]
        elif value > 6:
            keys_with_score_over_5.append(key)
    return keys_with_score_over_5


def generate_feature_prompt(webs, qtoken_list, verify_list=None):
    features_prompt = {}
    for key, q in qtoken_list.items():
        if verify_list is None or key in verify_list:
            web_index = q.pages[0]["page_number"]
            web_key = q.pages[0]["value"][0]
            example_value = []
            for leaf in webs[web_index - 1].tree:
                if leaf["key"] == web_key:
                    for leaf_index in range(1, len(leaf["value"])):
                        example_value.append(leaf["value"][leaf_index])
                        if leaf_index >= 2:
                            break
            pair = (q.annotation, example_value)
            features_prompt[key] = pair

    return features_prompt
