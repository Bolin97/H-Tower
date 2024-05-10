import random
import webpage
from gpt_api import formulate_gpt_question
from default_data import DEMAND, WEBPAGE, QTOKEN, read_token
from cal_average_length import cal
from token_select_strategy import select_token, find_qualify_value, remove_token, remove_page

# 用户对话版
webPages = webpage.webPages
token_averages, page_key2token = cal(webPages)  # 计算每一个token的item长度（平均值）
qtoken_list = read_token(token_averages)  # 完整的token列表，即待询问的token
# 待选页面
possible_pages = [WEBPAGE() for _ in range(len(webPages))]
final_demands = {}

if __name__ == '__main__':

    q_round = 0  # 当前轮次
    effective_round = 0  # 有效轮次
    kvalue = ""
    while kvalue != "需求结束" and len(qtoken_list) != 0:  # 只要possible_pages不是1
        # 选择要询问的token
        index, selected_qtoken = select_token(qtoken_list)
        token = selected_qtoken.token  # 获取token
        annotation = selected_qtoken.annotation  # 获取annotation
        print("当前询问的token为：", token)
        effective = False  # 该论询问是否获得有效信息
        while effective is False:
            # GPT生成问句
            question = formulate_gpt_question(token, annotation, "question")
            print("GPT：", question)
            q_round = q_round + 1  # 询问轮次+1
            # 用户输入关键词
            kvalue = input("user: ")
            if kvalue == "无" or kvalue == "":
                effective = True
                # 用户对此token无兴趣，不用再询问
                remove_token(qtoken_list, token)
                break
            elif kvalue == "需求结束":
                break
            # 本次回答是否有效，（网页是否有对应的value）并存储对于有效的网页
            if effective_round == 0:
                effective = find_qualify_value(possible_pages, final_demands, selected_qtoken, webPages, kvalue)
            else:
                effective = find_qualify_value(possible_pages, final_demands, selected_qtoken, possible_pages, kvalue)
            if effective:  # 有效问答
                effective_round = effective_round + 1
                # 用户已经回答对应的要求并且是有效的
                remove_token(qtoken_list, token)
                # 移除n-1轮中不满足第n轮要求的pages
                remove_page(possible_pages, selected_qtoken)
    # 对于possible_page
    qualify_page_numbers = [page.index for page in possible_pages if page.index is not None]
    for i in qualify_page_numbers:
        print("pageNumber:" + str(i), final_demands[i])
    print("conversation done")
