import copy
import json
import random

from tqdm import tqdm

from agent_data_process import spilt_score, generate_feature_prompt
from cal_average_length import cal
from default_data import CONVERSATION, EXPLICIT, read_domain, read_token, PROCESSINGTOKEN, DEMAND
from gpt_api import formulate_gpt_question
from llm_Infer import feature_value_infer, token_infer
from token_select_strategy import select_token, save_qualify_value, remove_from_demand
from user_simulator import expression_generation, requirement_judgment, \
    requirement_confirmation, user_choose_token, requirement_validation, check_verify_list
from webpage import load_webpages

# 需求相关
user_demand = DEMAND()  # 完整的用户需求列表 # user+agent
remain_demand = DEMAND()  # user没有说过的需求 # user
user_profiles = None  # 该轮需求对话中的user_profiles # user+agent

final_pagesAkv = {}  # 最终agent推测出的符合需求列表的网页 以及 用户需求在该网页的表现形式
user_said_demand = DEMAND()  # 从上一次verify到这一次，user认为自己提到了的feature # user
last_confirmed_demand = {}  # 最后一次verify之前agent认为自己确定了的feature # agent
processing_list = {}  # verify之前累计的需要处理的需求 # agent

# 用户需求真实的领域和类型
class_domain = None
class_type = None
# agent推测出来的domain和type
demands_domain = None
demands_type = None

qtoken_list = {}  # 初始是完整的token列表，即待询问的token，会随着对话轮次变化 # agent
origin_qtoken_list = {}  # 完整的token列表 # agent

webPages = []  # 该领域内的所有网页
possible_pages = []  # 待选页面（初始化为全部）
features_prompt = {}  # 可以被推理的token
conversations = []  # 对话列表
unwanted_token_times = 0  # agent询问用户不感兴趣的token的次数
interactions_rounds = 0  # 交互轮次


# 记录对话
def save_conversation(role, sentence, s_type, target_feature=None, user_tokens=None, user_values=None):
    global conversations
    global interactions_rounds
    con = CONVERSATION()
    con.role = role
    con.sentence = sentence
    con.type = s_type
    con.target_feature = target_feature  # 询问的目标token
    con.token = user_tokens  # 用户回答了哪些token
    con.value = user_values  # 用户期望的value
    conversations.append(con)
    interactions_rounds = interactions_rounds + 1


# 读取需求列表
def readDemand_and_judgeDomain(demand):
    global user_demand, class_type, class_domain, user_profiles, remain_demand
    user_demand.key = list(demand["need"].keys())
    user_demand.token = demand["features"]
    for key in user_demand.key:
        user_demand.value.append(demand["need"][key])
    # 用户列表的信息
    class_type = demand["demand_type"]
    class_domain = demand["domain"]
    user_profiles = demand["user_profile"]
    user_values = []
    # trigger_features是token
    for feature in demand["trigger_features"]:
        user_values.append(demand["need"][user_demand.key[user_demand.token.index(feature)]])
    # 将user_demand复制到remain_demand中
    remain_demand = copy.deepcopy(user_demand)
    # 将trigger中用户已经说过的token从remain_demand中去掉
    for said_token in demand["trigger_features"]:
        remove_from_demand(remain_demand, said_token)
    judgeDomain_Type(demand["trigger"], demand["trigger_type"], demand["trigger_features"], user_values)


# 推测domain和type
def infer_domain_type(sen, types, correct_type):
    option = []
    domain_or_class_type = ""
    while len(option) < 1:
        result, _ = formulate_gpt_question(None, sen, types, "domain_infer")
        option = spilt_score(result)
    # domain和type的澄清
    if len(option) > 1:
        confirm_question, question_type = formulate_gpt_question(None, option, sen, "domain_confirm")
        print("agent:", confirm_question)
        save_conversation("agent", confirm_question, question_type)
        domain_answer, answer_type = formulate_gpt_question(None, confirm_question, correct_type, "domain_answer",
                                                            option)
        print("user:", domain_answer)
        save_conversation("user", domain_answer, answer_type)
        for op in option:
            if op in domain_answer:
                domain_or_class_type = op
    else:
        domain_or_class_type = option[0]
    return domain_or_class_type


# trigger以及进行准备操作
def judgeDomain_Type(trigger_sen, trigger_type, user_tokens, user_values):
    global remain_demand, class_type, demands_domain, origin_qtoken_list
    # 读取domains和types
    print("\nuser:", trigger_sen)
    # 记录
    save_conversation("user", trigger_sen, "开始," + trigger_type, user_tokens, user_tokens, user_values)
    all_domains, all_types, path = read_domain()
    domains = ','.join(all_domains)
    # 判断领域
    global demands_domain, demands_type
    domain = infer_domain_type(trigger_sen, domains, class_domain)
    demands_domain = domain
    # 判断子类型
    index = all_domains.index(domain)
    domain_types = all_types[index]
    if len(domain_types) != 1:
        types = ','.join(domain_types)
        infer_type = infer_domain_type(trigger_sen, types, class_type)
        demands_type = infer_type
    else:
        demands_type = domain_types[0]

    global webPages, qtoken_list, possible_pages
    domain_path = path[index]
    # 该领域的所有网页
    webPages = load_webpages(domain_path, demands_domain)
    possible_pages = copy.deepcopy(webPages)
    # 操作子类
    token_averages, page_key2token, valid_tokens, TF, valid_page_numbers = cal(webPages, domain_path, demands_type)
    # 该子领域的所有token
    qtoken_list = read_token(token_averages, domain_path, TF, valid_tokens, valid_page_numbers)
    origin_qtoken_list = copy.deepcopy(qtoken_list)
    infer_and_judge(trigger_sen, trigger_type, user_tokens, None)


# 由于推理错误或者漏推而进行的重新推理
def re_infer(token, answer, process_info):
    global qtoken_list, last_confirmed_demand
    process_info.infer_value = feature_value_infer(qtoken_list, webPages, token, answer)
    while process_info.unsuccessful <= 4 or process_info.unsuccessful != -1 and process_info.interested is False:
        while process_info.infer_value is None and process_info.interested is not False:
            if process_info.interested is not True:
                result = clarify_token(answer, token, user_said_demand.token)
                if result == '反对':
                    process_info.interested = False
                    break
                else:
                    process_info.interested = True
            process_info.unsuccessful += 1
            process_info.infer_value, process_info.token_answer_type = assist(token, process_info, answer,
                                                                              qtoken_list[token])
        # 重新推理出值
        if process_info.infer_value is not None:
            if process_info.token_answer_type == EXPLICIT:
                last_confirmed_demand[token] = process_info.infer_value
                process_info.unsuccessful = -1
                process_info.interested = True
                del qtoken_list[token]
                break
            # 进行澄清
            else:
                correct = clarify(token, process_info, answer)
                # 澄清发现是错误的
                if not correct:
                    process_info.infer_value = None
                    # unsuccessful + 1在后续的步骤中，这里不需要重复写
                # 澄清发现是正确的
                else:
                    # value被确定了
                    last_confirmed_demand[token] = process_info.infer_value
                    process_info.unsuccessful = -1
                    process_info.interested = True
                    del qtoken_list[token]
                    break


# 验证环节(未解决的需求再解决 默认正确的需求进行确认 )
def verify():
    global class_type, processing_list, possible_pages, user_demand, \
        origin_qtoken_list, qtoken_list, user_said_demand, features_prompt
    # 列举已经完成的token
    init_verify_sen = "好的。您还觉得有其他方面需要改进吗？"
    print("以下是通过刚刚的对话我所了解的需求，请您核对以下的列表，如果有错误或者遗漏请及时告诉我。" + "\n")
    verify_list = []
    for token, value in processing_list.items():
        if value is not None:
            print(token + ":" + value)
            verify_list.append(token)  # 展示给用户看的列表

    # no_need_token, wrong_token_value, none_token_value+missed_token_value是user知道的
    # none_token_value是agent知道的
    no_need_token, wrong_token_value, none_token_value, missed_token_value = check_verify_list(user_said_demand,
                                                                                               verify_list,
                                                                                               processing_list)

    # none_token_value是确认有兴趣的token，在前面的操作中从qtoken list中去掉了,wrong_token_value不一定
    if len(no_need_token) > 0:
        for token in verify_list:
            if processing_list[token].unsuccessful != -1:
                if token not in list(qtoken_list.keys()):
                    qtoken_list[token] = copy.deepcopy(origin_qtoken_list[token])

        features_prompt = generate_feature_prompt(webPages, qtoken_list, verify_list)

        # 第一步：处理verify_list中用户不需要的token
        no_need_answer = requirement_validation(class_type, 'no_need', no_need_token)
        print("user:", no_need_answer)
        save_conversation("user", no_need_answer, "verify-多余的feature")
        infer_no_need_token = token_infer(no_need_answer, None, features_prompt)
        # 检查合法性
        for token in infer_no_need_token:
            if token not in verify_list:
                infer_no_need_token.remove(token)
            else:
                # 不进行确认，直接删除，如果有错误，等到最后的verify再解决
                if processing_list[token].interested is not True:
                    del processing_list[token]
        agent_sen, _ = formulate_gpt_question(None, init_verify_sen, None, "ask_for_verify")
        print("agent:", agent_sen)
        save_conversation("agent", agent_sen, "verify-多余的feature")
    if len(wrong_token_value) > 0:
        # 第二步：处理verify list中错误的value（一次性说出来）
        wrong_value_answer = requirement_validation(class_type, 'wrong_value', wrong_token_value,
                                                    processing_list)
        print("user:", wrong_value_answer)
        save_conversation("user", wrong_value_answer, "verify-错误value")
        infer_wrong_value_token = token_infer(wrong_value_answer, None, features_prompt)
        # 检查合法性,由于第一步的限制，不会将用户感兴趣的删除
        for token in infer_wrong_value_token:
            if token not in verify_list:
                infer_wrong_value_token.remove(token)
            else:
                if token in list(processing_list.keys()):
                    if processing_list[token].unsuccessful != -1:
                        processing_info = processing_list[token]
                        processing_info.dialog_history.append(wrong_value_answer)
                        # 重新推理，需要correct
                        # token肯定是被需要的
                        re_infer(token, wrong_value_answer, processing_list[token])
                    else:
                        infer_wrong_value_token.remove(token)
                        continue
                else:
                    result = clarify_token(wrong_value_answer, token, wrong_token_value)
                    if result == "反对":
                        # infer错误
                        infer_wrong_value_token.remove(token)
                        continue
                    else:
                        # 第一步中被删除了
                        processing_token = PROCESSINGTOKEN()
                        processing_token.value_type = qtoken_list[token].token_type
                        processing_token.dialog_history.append(wrong_value_answer)
                        processing_list[token] = processing_token
                        # token肯定是被需要的
                        re_infer(token, wrong_value_answer, processing_token)
        agent_sen, _ = formulate_gpt_question(None, init_verify_sen, None, "ask_for_verify")
        print("agent:", agent_sen)
        save_conversation("agent", agent_sen, "verify-错误的value")

    if len(no_need_token) > 0 or len(missed_token_value) > 0:
        # 第三步： 用户提到过的但是没在verify列表中看见
        missed_all = none_token_value + missed_token_value
        missed_answer = requirement_validation(class_type, 'missed_token', missed_all, processing_list)
        print("user:", missed_answer)
        save_conversation("user", missed_answer, "verify-遗漏feature")
        prompt_list = copy.deepcopy(qtoken_list)
        for token in verify_list:
            del prompt_list[token]
        # 这样推测出来的token就不会于verify list有重合了
        features_prompt = generate_feature_prompt(webPages, prompt_list)
        infer_missed_tokens = token_infer(missed_token_value, none_token_value, features_prompt)
        for token in infer_missed_tokens:
            if token not in list(prompt_list.keys()):
                # 合法性检查
                infer_missed_tokens.remove(token)
            else:
                if token in none_token_value:
                    processing_info = processing_list[token]
                    processing_info.dialog_history.append(missed_answer)
                    # 是之前推测过的token
                    # token肯定是被需要的
                    re_infer(token, missed_answer, processing_list[token])
                else:
                    # 没有推测过的token
                    processing_token = PROCESSINGTOKEN()
                    processing_token.value_type = qtoken_list[token].token_type
                    # 将回答存入对话历史中
                    processing_token.dialog_history.append(missed_answer)
                    processing_list[token] = processing_token
                    # token不一定是被需要的
                    re_infer(token, missed_answer, processing_token)

    record_infer_feature()
    # 重置
    processing_list = {}
    user_said_demand = DEMAND()


def record_infer_feature():
    global processing_list
    # 逐个获取成功infer的kv的selected_qtoken并记录
    # processing_list中的token一定是用户提到过的(很小的概率是用户没提到但是模型推理出来value，并且value在answer中出现了)
    for s_token in list(processing_list.keys()):
        # 推理成功
        processing_info = processing_list[s_token]
        if processing_info.unsuccessful == -1:
            # 用户已经回答对应的要求并且是有效的
            # 从待询问的token列表中删除
            last_confirmed_demand[s_token] = processing_info.infer_value
            # 没有确定是正确的就被抛弃了
            if s_token in list(qtoken_list.keys()):
                del qtoken_list[s_token]


# 模拟对话
def simulation_process():
    global qtoken_list, user_demand, remain_demand
    is_end = False
    while is_end is False and len(qtoken_list) != 0:
        # 在对话期间的verify,待确定的processing_list数量>=3
        if len(processing_list.keys()) >= 3:
            verify()
        way = random.randint(0, 5)
        # 先提问当前网页中的required
        required_tokens_union = set()
        for webpage in webPages:
            required_tokens_union.update(webpage.required_tokens)
        required_token = list(required_tokens_union)
        if len(required_token) > 0:
            for re in required_token:
                # required还没被问过
                if re in remain_demand.token:
                    way = 1
        # 第一轮当用户trigger中没有需求的时候不要开放询问
        if way < 5 or len(user_demand.token) == len(remain_demand.token):
            is_end = one_token_process()
        else:
            is_end = open_ended_process()
    # 最后的追问和验证环节
    verify()


# 开放式询问
def open_ended_process():
    global class_type, demands_domain, qtoken_list, unwanted_token_times, user_said_demand
    unwanted_token_times = 0
    # 剩余的未问过的token
    max_len = len(qtoken_list)
    random_num = min(3, max_len)
    random_tokens = random.sample(list(qtoken_list.keys()), random_num)
    random_annotations = []
    # 获取示例的token和annotation
    for token in random_tokens:
        random_annotations.append(qtoken_list[token].annotation)

    # agent
    question, question_type = formulate_gpt_question(demands_type, random_tokens, random_annotations,
                                                     'open_ended_question')
    print("\n开放式询问:")
    print("agent:", question)
    save_conversation("agent", question, question_type)
    # user
    answer, answer_type, user_tokens, user_values, user_kvs = user_choose_token(class_type, question, remain_demand,
                                                                                qtoken_list, user_profiles)
    for index, token in user_tokens:
        if token not in user_said_demand.token:
            user_said_demand.token.append(token)
            user_said_demand.value.append(user_values[index])
        remove_from_demand(remain_demand, token)
    print("user:", answer)
    save_conversation("user", answer, answer_type, user_tokens, user_tokens, user_values)
    # 判断用户需求是否结束
    demand_end, _ = formulate_gpt_question(None, question, answer, "is_end")
    if "B" in demand_end:
        infer_and_judge(answer, answer_type, user_tokens, None)
        return False
    else:
        return True


# 一次询问一个token
def one_token_process():
    global qtoken_list, conversations, unwanted_token_times, class_type, demands_domain, user_profiles, user_said_demand

    # 选择要询问的token，select_token是询问策略
    q_token = select_token(qtoken_list)
    selected_qtoken = qtoken_list[q_token]  # 获取token
    annotation = selected_qtoken.annotation  # 获取annotation

    example_value = features_prompt[q_token][1]
    print("\n当前询问的token为：", q_token)
    # GPT生成问句
    question, gpt_type = formulate_gpt_question(demands_type, q_token, annotation, "question", example_value)
    print("agent：", question)
    save_conversation("agent", question, gpt_type + "询问", q_token)
    # 用户回答
    # 若选择的token不在需求列表中(直接回答了)
    if q_token not in remain_demand.token:
        answer, _ = formulate_gpt_question(demands_type, question, None, 'no_interest')
        print("user:", answer)
        unwanted_token_times = unwanted_token_times + 1
        save_conversation("user", answer, "无兴趣回答")
        # 用户对此token无兴趣，不用再询问
        del qtoken_list[q_token]
        # 当多次没有选中用户想要的token时
        if unwanted_token_times == 2:
            is_end = open_ended_process()
            return is_end

    # 若选择的token在需求列表中
    else:
        unwanted_token_times = 0
        demand_index = remain_demand.token.index(q_token)
        # 该token对应的value（需求列表中对应的）
        kvalue = remain_demand.value[demand_index]
        # 生成回答（可以带1-2个未问到的）
        answer, answer_type, user_tokens, user_values = expression_generation(question, q_token, kvalue,
                                                                              remain_demand, "question", qtoken_list,
                                                                              user_profiles)
        for index, token in user_tokens:
            if token not in user_said_demand.token:
                user_said_demand.token.append(token)
                user_said_demand.value.append(user_values[index])
            remove_from_demand(remain_demand, token)
        user_kvs = {}
        for index, u_token in enumerate(user_tokens):
            user_kvs[u_token] = user_values[index]
        print("user:", answer)
        save_conversation("user", answer, answer_type + "回答", q_token, user_tokens, user_values)
        # 推理
        infer_and_judge(answer, answer_type, user_tokens, q_token)
    return False


# 对推理的token向用户进行确认
def clarify_token(sen, infer_token, user_tokens):
    infer_token_qtoken = qtoken_list[infer_token]
    infer_token_annotation = infer_token_qtoken.annotation
    question, question_type = formulate_gpt_question(None, sen, infer_token, "clarify_token", infer_token_annotation)
    save_conversation("agent", question, question_type, infer_token)
    print("agent(token_clarify):", question)
    if infer_token in user_tokens:
        answer, _ = formulate_gpt_question(None, question, None, "agree")
    else:
        answer, _ = formulate_gpt_question(None, question, None, "disagree")
    print("user:", answer)
    save_conversation("user", answer, None, infer_token)

    emotion, _ = formulate_gpt_question(None, answer, None, None)
    return emotion


# 推理及后续处理
def infer_and_judge(answer, answer_type, user_tokens, certain_token):
    global qtoken_list, remain_demand, features_prompt, processing_list, user_demand, user_profiles
    features_prompt = generate_feature_prompt(webPages, qtoken_list)
    # 推断answer涉及的所有token
    infer_tokens = token_infer(answer, certain_token, features_prompt)
    # 筛选不合法的infer_tokens
    for infer_token in infer_tokens:
        if infer_token not in list(qtoken_list.keys()):
            infer_tokens.remove(infer_token)

    # 回答语句中不包含需求（完全模糊触发句）
    if "None" in infer_tokens:
        return

    # 保存推断出来的token和value的信息，以及在文件中的信息
    for token in infer_tokens:
        # 表示是当前这几轮在处理的token
        processing_token = PROCESSINGTOKEN()
        processing_token.value_type = qtoken_list[token].token_type
        infer_value, options = feature_value_infer(qtoken_list, webPages, token, answer)
        processing_token.infer_value = infer_value
        processing_token.options = options
        processing_token.token_answer_type = answer_type
        # 将回答存入对话历史中
        processing_token.dialog_history.append(answer)
        processing_list[token] = processing_token

    # 对于回答中的每一个token
    for index, infer_token in enumerate(infer_tokens):
        process_info = processing_list[infer_token]
        correct_token = None
        while process_info.unsuccessful < 4 and correct_token is not False:
            # 用户在回答中明确提到了token
            if infer_token in answer:
                correct_token = True
            # 辅助和重述阶段
            while process_info.unsuccessful < 4 and process_info.infer_value is None and correct_token is not False:
                # 为某一次回答中的每一个token配对一个value（value是网页中有的）
                # unsuccessful是推测出None的次数
                # 并不要求infer_value是正确的，只要有就行
                # 可能这个token并不是用户所希望的
                if infer_token != certain_token:
                    if (correct_token is not True) or (
                            process_info.unsuccessful == 1 and process_info.interested is not True):
                        # 第二个判断条件是为了纠正回答中提到token就认为有兴趣的潜在问题
                        # 只询问【当前这句话】是不是有需求
                        result = clarify_token(answer, infer_token, user_tokens)
                        if result == "反对":
                            # 只从需要待推测的token列表中删除（可能之后有要求）
                            del processing_list[infer_token]
                            correct_token = False
                            break
                        else:
                            correct_token = True
                            process_info.interested = True
                # 未确认value(infer_value不在可选value中，或根本就没推测出来)
                # infer不成功，需要辅助确认或者用户重述
                process_info.unsuccessful = process_info.unsuccessful + 1
                # 如果这个值是因为存在于选项中被用户选择了，则大概率是正确的
                process_info.infer_value, process_info.token_answer_type = assist(infer_token, process_info, answer,
                                                                                  qtoken_list[infer_token])
                if process_info.infer_value is not None:
                    process_info.unsuccessful = -1
                    process_info.interested = True
    for s_token in list(processing_list.keys()):
        # 将确定的用户在此次对话中说到的过token从待询问的列表中删除（保证之后不会再问到）,即使当前推出来的是None，后面verify时用户会说
        process_info = processing_list[s_token]
        # 只是防止token infer时的干扰，所以对于只是推出值但是没有完全肯定的不删除
        if process_info.interested:
            del qtoken_list[s_token]


# agent辅助用户进行确认，带值询问
def assist(feature, process_info, requirement, qtoken):
    global class_type, demands_domain
    if qtoken.token_type == "选项":
        # 随机选择 5 个值
        if len(process_info.options) > 5:
            examples = random.sample(process_info.options, 5)
        else:
            examples = process_info.options
        # 去掉被问过的
        process_info.options = [example for example in process_info.options if example not in examples]
        matching_qtoken = qtoken_list[feature]
        annotation = matching_qtoken.annotation
        # agent
        retell_probability = random.randint(0, 1)  # 询问语句中可能需要用户重述
        if retell_probability == 0:
            question, gpt_type = formulate_gpt_question(demands_domain, feature, requirement, "assist_options",
                                                        examples,
                                                        annotation)
        else:
            question, gpt_type = formulate_gpt_question(demands_domain, feature, requirement, "assist_and_retell",
                                                        examples,
                                                        annotation)
        print("agent(assist):", question)
        process_info.dialog_history.append("agent:" + question)
        save_conversation("agent", question, gpt_type, feature)
        # user
        value = process_info.user_value
        answer, answer_type = requirement_confirmation(class_type, question, value, "options")
        association_value = None
        a_type = None
        if answer_type == "存在于选项中":
            print("user:", answer)
            save_conversation("user", answer, answer_type)
            association_value, _ = feature_value_infer(qtoken_list, webPages, feature, answer)
            process_info.dialog_history.append("user:" + question)
            a_type = EXPLICIT
        else:
            # 不存在于选项中，则有一定几率重述
            if retell_probability == 1:
                # 重述后会再一次被infer
                association_value, a_type = retell(question, feature, value, answer, requirement, process_info)
            else:
                # 不重述
                print("user:", answer)
                process_info.dialog_history.append("user:" + question)
                save_conversation("user", answer, answer_type)
        return association_value, a_type
    else:
        results, _ = formulate_gpt_question(None, process_info.options, None, "sort")  # 将数值型的value排序
        start_index = results.find("['")
        end_index = results.find("']")
        sorted_examples = results[start_index + 2:end_index].split("', '")
        # 4-5次即可
        while len(sorted_examples) > 0:
            index = int((len(sorted_examples) - 1) / 2)
            middle_value = sorted_examples[index]
            question, gpt_type = formulate_gpt_question(demands_type, feature, process_info.dialog_history,
                                                        "assist_numeric", middle_value)
            process_info.dialog_history.append("agent：" + question)
            print("agent(assist numeric):", question)
            save_conversation("agent", question, gpt_type, feature)
            value = process_info.user_value
            answer, answer_type = requirement_confirmation(class_type, question, value, "numeric")
            process_info.dialog_history.append("user：" + answer)
            print("user:", answer)
            save_conversation("user", answer, answer_type)
            numeric_result, _ = formulate_gpt_question(None, question, answer, "numeric_result")
            if "A" in numeric_result:
                for i in range(index, len(sorted_examples)):
                    sorted_examples.remove(sorted_examples(i))
            elif "C" in numeric_result:
                for i in range(0, index):
                    sorted_examples.remove(sorted_examples(i))
            elif "B" in numeric_result:
                return middle_value, sorted_examples, None
        process_info.options = sorted_examples
        return None, None


# 重述
def retell(question, token, value, answer, requirement, process_info):
    global class_type, user_profiles
    # user
    retell_answer, answer_type, user_tokens, user_values = expression_generation(question, token, value,
                                                                                 requirement, "retell", qtoken_list,
                                                                                 user_profiles)
    print("user:", answer + "(重述)" + retell_answer)
    process_info.dialog_history.append("user:" + answer + retell_answer)
    save_conversation("user", answer + retell_answer, answer_type + "回答", token, user_tokens, user_values)
    infer_value, _ = feature_value_infer(qtoken_list, webPages, token, retell_answer)
    return infer_value, answer_type


# agent向用户询问以澄清
def clarify(feature, process_info, sentence):
    # 该语句是澄清语句,澄清用户回答的所有token
    global conversations, class_type, demands_domain
    # agent
    user_value = process_info.user_value
    infer_value = process_info.infer_value
    clarify_question, sen_type = formulate_gpt_question(demands_type, feature, infer_value, "clarify", sentence)
    print("agent(clarify)：", clarify_question)
    process_info.dialog_history.append("agent:" + clarify_question)
    save_conversation("agent", clarify_question, sen_type, feature, None, infer_value)
    # user
    result, answer, answer_type = requirement_judgment(class_type, user_value, infer_value, clarify_question)
    print("user:", answer)
    process_info.dialog_history.append("user:" + answer)
    save_conversation("user", answer, answer_type)
    return result


# 将对话写入文件
def write_to_file():
    global conversations
    conversations_json = []
    for con in conversations:
        conversation_dict = {
            "role": con.role,
            "sentence": con.sentence,
            "type": con.type,
            "target_feature": con.target_feature,
            "token": con.token,
            "value": con.value
        }
        conversations_json.append(conversation_dict)

    # 将 conversations_json 列表保存到 JSON 文件中
    with open("./data/conversations.json", "w", encoding="utf-8") as f:
        json.dump(conversations_json, f, indent=4, ensure_ascii=False, )


def run():
    with open("data/demands_0.json", "r", encoding="utf-8") as file:
        demands = json.load(file)
        # 对用户需求列表中的每一个列表
        # 存储key token和value
    for demand in tqdm(demands):
        # 判断用户需求的领域，生成trigger
        readDemand_and_judgeDomain(demand)
        # 开始模拟
        simulation_process()
        # 写入文件
        write_to_file()
        global possible_pages
        # 将token-value变成key-value并与网页页面对应
        for token in list(last_confirmed_demand.keys()):
            selected_qtoken = origin_qtoken_list[token]
            save_qualify_value(final_pagesAkv, selected_qtoken, possible_pages, last_confirmed_demand[token])
        print("\n用户原始需求:")
        for i in range(0, len(user_demand.key)):
            print(user_demand.key[i], ":", user_demand.value[i])
        print("\n符合要求的页面:")
        for index, _ in final_pagesAkv.items():
            # if len(final_pagesAkv[i]) == len(user_demand.key):
            print("pageNumber:" + str(index), final_pagesAkv[index])
        global interactions_rounds
        print("交互轮次：", interactions_rounds / 2)
        print("conversation done")


if __name__ == '__main__':
    run()
