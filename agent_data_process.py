# 对可能的领域打分
def spilt_score(data):
    # 初始化空字典
    scores = {}
    # 按行分割文本
    lines = data.split('\n')
    # 遍历每一行
    for line in lines:
        # 忽略空行
        if line.strip() == '':
            continue
        # 使用冒号分割键值对
        if ":" in line:
            key, value = line.split(':')
            # 将分数转换为整数
            try:
                # 尝试将值转换为整数
                score = int(value)
                # 将键值对添加到字典中
                scores[key.strip()] = score
            except ValueError:
                continue
    keys_with_score_over_5 = [key for key, value in scores.items() if value > 5]
    keys_with_score_over_5.sort(reverse=True)
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
