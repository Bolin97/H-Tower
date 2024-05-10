import json
import random
import webpage
from gpt_api import formulate_gpt_question
from default_data import DEMAND, QTOKEN, read_token, WEBPAGE, CONVERSATION
from cal_average_length import cal
from token_select_strategy import select_token, find_qualify_value, remove_token, remove_page

user_demand = DEMAND()
webPages = webpage.webPages
token_averages, page_key2token = cal(webPages)  # 计算每一个token的item长度（平均值）
qtoken_list = read_token(token_averages)  # 完整的token列表，即待询问的token
# 待选页面
possible_pages = [WEBPAGE() for _ in range(len(webPages))]
final_demands = {}

if __name__ == '__main__':
    remain_demand = DEMAND()
    # 构建需求列表
    page_num = random.randint(0, len(webPages) - 1)  # 随机选择一个页面
    user_demand.page_number = page_num + 1  # 网页编号
    key_num = random.randint(1, len(webPages[page_num].keys) - 1)  # 随机选择key的数量
    selected_tree = random.sample(webPages[page_num].tree, key_num)
    for kv in selected_tree:
        user_demand.key.append(kv["key"])
        selected_value = "不限"
        while selected_value == "不限":
            selected_value = random.sample(kv["value"], 1)[0]
        user_demand.value.append(selected_value)
    print("\n需求列表：")
    for i in range(0, len(user_demand.key)):
        user_demand.token.append(page_key2token.get((user_demand.page_number, user_demand.key[i])))
        print(user_demand.key[i] + ":" + user_demand.value[i])
    remain_demand = user_demand

    q_round = 0  # 当前轮次
    effective_round = 0  # 有效轮次
    conversations = []  # 对话列表
    while effective_round != len(user_demand.key) and len(qtoken_list) != 0:  # 只要remain_demand中仍然有需求
        # 选择要询问的token，询问策略的选择
        index, selected_qtoken = select_token(qtoken_list)
        token = selected_qtoken.token  # 获取token
        annotation = selected_qtoken.annotation  # 获取annotation
        web_index = selected_qtoken.pages[0]["page_number"]
        web_key = selected_qtoken.pages[0]["value"][0]
        # 获取key的示例，用于prompt生成问句
        example_value = []
        for leaf in webPages[web_index - 1].tree:
            if leaf["key"] == web_key:
                for i in range(1, len(leaf["value"])):
                    example_value.append(leaf["value"][i])
                    if i >= 2:
                        break
        print("\n当前询问的token为：", token)
        effective = False  # 该论询问是否获得有效信息
        while effective is False:
            # 对话
            conversation = CONVERSATION()
            # GPT生成问句
            question = formulate_gpt_question(token, annotation, "question", example_value)
            # 判断问句是否有效（暂时没有判断方法）

            print("GPT：", question)
            conversation.token = token
            conversation.questions.append(question)
            q_round = q_round + 1  # 询问轮次+1

            # 用户回答
            # 若选择的token不在需求列表中
            if token not in remain_demand.token:
                effective = True
                answer = formulate_gpt_question(question, "无", "answer")
                print("user:", answer)
                conversation.answers.append(answer)
                # 用户对此token无兴趣，不用再询问
                remove_token(qtoken_list, token)
                conversations.append(conversation)
                break
            # 本次提问是否有效，（网页是否有对应的value）并存储对于有效的网页
            demand_index = remain_demand.token.index(token)
            kvalue = remain_demand.value[demand_index]
            # 生成回答
            answer = formulate_gpt_question(question, kvalue, "answer")
            print("user:", answer)
            conversation.value = kvalue
            conversation.answers.append(answer)

            if effective_round == 0:  # 第一个回合过后的剪枝
                effective = find_qualify_value(possible_pages, final_demands, selected_qtoken, webPages, kvalue)
            else:  # 其余回合过后的剪枝
                effective = find_qualify_value(possible_pages, final_demands, selected_qtoken, possible_pages, kvalue)
            if effective:  # 有效问答
                effective_round = effective_round + 1
                # 用户已经回答对应的要求并且是有效的
                remove_token(qtoken_list, token)
                # 移除n-1轮中不满足第n轮要求的pages
                remove_page(possible_pages, selected_qtoken)
            conversations.append(conversation)
    # 打印符合用户需求的page
    qualify_page_numbers = [page.index for page in possible_pages if page.index is not None]
    print("\n用户原始需求:")
    for i in range(0, len(user_demand.key)):
        print(user_demand.key[i], ":", user_demand.value[i])
    print("\n符合要求的页面:")
    for i in qualify_page_numbers:
        if len(final_demands[i]) == len(user_demand.key):
            print("pageNumber:" + str(i), final_demands[i])
    print("\n询问轮次:", q_round)
    print("conversation done")
    # 写入文件
    conversations_data = []
    for conv in conversations:
        conv_data = {
            "token": conv.token,
            "value": conv.value,
            "conversation": []
        }

        for i in range(len(conv.questions)):
            qa_pair = {
                "question": conv.questions[i],
                "answer": conv.answers[i]
            }
            conv_data["conversation"].append(qa_pair)

        conversations_data.append(conv_data)

    # 将数据写入到JSON文件中
    with open('../data/conversations.json', 'w', encoding="utf-8") as file:
        file.write(json.dumps(conversations_data, ensure_ascii=False, indent=2))
