import copy
import json
import math
import random
import time
import re
import ast

import pypinyin
from pypinyin import lazy_pinyin
from tqdm import tqdm
from agent_data_process import spilt_score, generate_feature_prompt
from cal_average_length import cal
from default_data import CONVERSATION, EXPLICIT, read_domain, read_token, PROCESSINGTOKEN, DEMAND, USE_USERPROFILE
from gpt_api import formulate_gpt_question
from llm_Infer import feature_value_infer_llm, token_infer
from token_select_strategy import select_token, save_qualify_value, remove_from_demand
from user_simulator import expression_generation, requirement_judgment, \
    requirement_confirmation, user_choose_token, requirement_validation, check_verify_list, infer_token_judgement, \
    retell_generator
from webpage import load_webpages

# 对话轮次定义
# episode: 完整对话过程
# step: 用户与agent一次交互

# 需求相关变量
user_demand = DEMAND()  # 完整用户需求
remain_demand = DEMAND()  # 未提及需求
user_profiles = {}  # 当前对话用户画像
profile_list = {}  # 用户画像映射
final_pagesAkv = {}  # 最终匹配网页及需求表现
user_said_demand = DEMAND()  # 用户自认已提需求
last_confirmed_demand = {}  # 上次验证前确认需求

processing_list = {}  # verify之前累计的agent认为需要处理的需求
user_tokennum=0
usertoken=[]
# 需求领域和类型
class_domain = None # 真实领域
class_type = None # 真实类型
demands_domain = None # 推测领域
demands_type = None # 推测类型
qtoken_list = {}  # 待询问token列表
origin_qtoken_list = {}  # 原始token列表
webPages = []  # 当前领域网页
possible_pages = []  # 待选网页
features_prompt = {}  # token推理提示信息
unwanted_token_times = 0  # agent询问用户不感兴趣的token的次数
interactions_rounds = 0  # 交互轮次
current_conversations = [] #存储当前需求对话的列表
global token_count
global current_user_type
global umatch
import math

MAX_DIALOGUE_ROUNDS = 10  # 设置最大对话轮数限制

class UserType:
    def __init__(self, type_name, token_long=150, k=2.55, x0=1.55, alpha=0.88, beta=0.88, lambda_=2.25):
        self.type_name = type_name
        self.total_tokens = 0
        self.expected_needs = 0
        self.matched_needs = 0
        self.token_long = token_long  # 指数型用户的token长度阈值
        self.k = k  # 指数型用户的斜率参数
        self.x0 = x0  # 指数型用户的中点参数
        self.alpha = alpha  # 期望型用户的收益参数
        self.beta = beta  # 期望型用户的损失参数
        self.lambda_ = lambda_  # 期望型用户的损失厌恶系数
        self.matched_keys_count = 0
        self.unmatched_keys_count = 0
        self.user_features = {}

    def value_function(self, x):
        """
        计算价值函数V(x)。
        :param x: 收益情况为匹配键的数量，损失情况为不匹配键的数量
        :return: 感知价值V(x)
        """
        if x >= 0:  # 收益情况
            return x ** self.alpha
        else:  # 损失情况
            return -self.lambda_ * (-x) ** self.beta

    def should_end_conversation(self):
        if self.type_name == "exponential":
            normalized_x = self.total_tokens / self.token_long
            f_x = 1 / (1 + math.exp(-self.k * (normalized_x - self.x0)))

            return f_x > 0.8,f_x
        elif self.type_name == "expectation":
            x = self.unmatched_keys_count
            x=x/4
            v_x = self.value_function(x)
            max_value = 1
            min_value = -1 * 2.25
            normalized_v_x = (v_x - min_value) / (max_value - min_value)
            return normalized_v_x < 0.2,normalized_v_x
        return False,0

    def update_tokens(self, token_count):
        self.total_tokens += token_count

    def update_needs(self, expected, matched):
        self.expected_needs = expected
        self.matched_needs = matched

    def update_key_counts(self, token):
        """
        更新匹配和不匹配键的计数
        :param token: 正在询问/处理的token
        :param is_in_features: 该token是否存在于用户特征中
        """
        self.unmatched_keys_count += token


    def set_user_features(self, features):
        """
        设置用户特征以供参考
        :param features: 用户特征字典
        """
        self.user_features = features

def extract_feature_values(agent_question,annotation, user_answer, target_features):
    """
    从用户回答中提取特征值

    参数:
        agent_question: agent提问
        user_answer: 用户回答
        target_features: 目标特征

    返回:
        特征值字典
    """
    # 确保target_features是列表
    if target_features is None:
        return None
    # 调用大模型进行分析
    result, _, _, _ = formulate_gpt_question(target_features, agent_question, user_answer, "extract_feature_values",extra_1=annotation)
    try:
        # 尝试解析返回的JSON
        extracted_values = json.loads(result)
        return extracted_values
    except json.JSONDecodeError:
        # 如果解析失败，尝试从文本中提取JSON部分
        try:
            # 查找可能的JSON部分
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                extracted_values = json.loads(json_str)
                return extracted_values
        except:
            pass

        # 如果仍然失败，返回空字典
        print("无法从大模型回答中提取JSON格式的特征值")
        return {}
# 记录对话
def save_conversation(role, sentence, s_type, target_feature=None, user_tokens=None, user_values=None,think_process=None,tolerance=None,prospect=None,annotation=None):
    """
    记录对话信息。

    Args:
        role (str): 对话角色，"agent" 或 "user"。
        sentence (str): 对话内容。
        s_type (str): 对话类型。
        target_feature (str, optional): 询问的目标token。 Defaults to None.
        user_tokens (list, optional): 用户回答的token列表。 Defaults to None.
        user_values (list, optional): 用户回答的value列表。 Defaults to None.
        think_process (str, optional): 思考过程。 Defaults to None.
        tolerance (float, optional): 容忍度。 Defaults to None.
        prospect (float, optional): 前景。 Defaults to None.
    """
    global interactions_rounds,current_conversations, user_value
    interactions_rounds = interactions_rounds + 1
    conversation = CONVERSATION()
    conversation.role = role
    if think_process is not None:
        conversation.think_process= think_process
    conversation.sentence = sentence
    conversation.s_type = s_type
    conversation.target_feature = target_feature
    if tolerance is not None:
        conversation.tolerance = tolerance
    if prospect is not None:
        conversation.prospect = prospect

    # 当角色是用户且有目标特征时，使用大模型分析回答中的value
    if role == "user" and target_feature is not None and len(current_conversations) > 0:
        # 获取最近的agent问题
        agent_question = ""
        for conv in reversed(current_conversations):
            if conv.role == "agent":
                agent_question = conv.sentence
                break
        user_values = []
        # 如果找到了agent问题，调用大模型分析
        extracted_values = extract_feature_values(agent_question,annotation, sentence, user_tokens)

        # 如果成功提取到值，更新user_tokens和user_values
        if extracted_values and len(extracted_values) > 0:

            # 将提取到的特征值添加到列表中
            for feature, value in extracted_values.items():
                if value ==None:
                    value=''
                user_values.append(value)


    if user_tokens is not None:
        for i,n in zip(user_tokens,range(len(user_tokens))):
            if user_values is not None:
                conversation.feature_value[i]=user_values[n]
            else:
                conversation.feature_value[i]=user_values
    current_conversations.append(conversation)



# 读取需求列表
def readDemand_and_judgeDomain(demand):
    """
    读取用户需求，并判断领域和类型。

    Args:
        demand (dict): 用户需求字典，包含 "need", "features", "trigger", "trigger_type", "trigger_features", "user_profile", "demand_type", "domain"。
    """
    global user_demand, class_type, class_domain, user_profiles, remain_demand, current_user_type, user_tokennum
    # 读取需求列表的信息
    # feature (token)对应的真实key, token是我们用来询问的关键词，真实的key是这个token在网页中的表示词
    user_demand.key = list(demand["need"].keys()) # 获取需求对应的key列表
    # feature
    user_demand.token = demand["features"] # 获取需求feature (token) 列表
    # value
    for key in user_demand.key:
        user_demand.value.append(demand["need"][key]) # 获取需求value列表
    # 用户列表的信息
    class_type = demand["demand_type"] # 获取用户需求类型 (真实类型)
    class_domain = demand["domain"] # 获取用户需求领域 (真实领域)
    user_profiles = demand["user_profile"] # 获取用户画像信息

    # 如果是expectation类型的用户，设置用户特征
    if current_user_type and current_user_type.type_name == "expectation":
        current_user_type.set_user_features(demand["need"])

    user_values = []
    # trigger_feature对应的value (触发特征对应的value)
    for feature in demand["trigger_features"]:
        if feature in user_demand.token:
            user_values.append(demand["need"][user_demand.key[user_demand.token.index(feature)]]) # 获取触发特征对应的value
    # 将user_demand复制到remain_demand中
    remain_demand = copy.deepcopy(user_demand) # 复制用户需求到剩余需求列表
    # 将trigger中的token从remain_demand中去掉 (移除已在trigger中提及的需求)
    for said_token in demand["trigger_features"]:
        remove_from_demand(remain_demand, said_token) # 从剩余需求中移除已提及的token
    # 从trigger中判断领域
    # trigger是用户说的第一句话，用来触发对话
    judgeDomain_Type(demand["trigger"], demand["trigger_type"], demand["trigger_features"], user_values) # 根据trigger判断领域和类型

def convert_punctuation(text):
    """
    中文标点转英文标点

    参数:
        text: 输入文本

    返回:
        转换后文本
    """
    # 中英文标点映射表
    punctuation_map = {
        '，': ',', '。': '.', '！': '!', '？': '?',
        '“': '"', '”': '"', '‘': "'", '’': "'",
        '（': '(', '）': ')', '【': '[', '】': ']',
        '、': ',', '；': ';', '：': ':', '《': '<', '》': '>'
    }
    # 替换所有中文标点
    for cn, en in punctuation_map.items():
        text = text.replace(cn, en)
    return text

def full_to_half(text):
    """
    全角字符转半角

    参数:
        text: 输入文本

    返回:
        转换后文本
    """
    result = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:  # 判断是否为全角字符 (判断是否为全角字符范围)
            result.append(chr(code - 0xFEE0)) # 转换为半角字符
        else:
            result.append(char)
    return ''.join(result)

def full_convert(text):
    """
    全角字符和标点转换

    参数:
        text: 输入文本

    返回:
        转换后文本
    """
    # 1. 转换全角英文/数字
    text = full_to_half(text)
    # 2. 转换中文标点
    text = convert_punctuation(text)
    return text

# 推测domain和type
def infer_domain_type(sen, types, infe,class_type):
    """
    推断领域或类型。

    Args:
        sen (str): 触发句。
        types (list): 候选领域或类型列表。
        infe (str): 推断类型，"domain_infer" 或 "domain_confirm"。
        class_type (str): 真实的用户需求类型，用于辅助推断。

    Returns:
        str: 推断出的领域或类型。
    """
    option = [] # 存储推断结果
    correct_type=["书籍","影视","写字楼","车辆","住房"] # 预定义的正确类型列表
    sen1=''
    if infe=='domain_infer': # 领域推断
        while 1: # 循环直到推断出有效领域
            a=0
            result, _ ,_,_= formulate_gpt_question(None, sen+sen1, correct_type, infe) # 调用GPT接口进行领域推断
            result = re.sub(r"\s+", "", result) # 去除结果中的空格
            print("domain_infer",result)
            for i in correct_type: # 检查推断结果是否在预定义的类型列表中
                if i==result:
                    option.append(result) # 添加到推断结果列表
                    a=1
                    break
            if a==1: # 推断成功则跳出循环
                break
            else: # 推断失败，进行澄清
                result, answer_type,think_process,ntoken = formulate_gpt_question(None, sen, correct_type, 'domain_question') # 询问用户领域问题
                print('agent:',result)
                usertoken = list(set(user_demand.key) - set(remain_demand.key))
                a=f'当前已进行{int(len(current_conversations)//2)}轮对话，得到{len(usertoken)}项用户需求:{usertoken},'
                think_process=a+think_process
                save_conversation("agent", result, "domaininfo_question",correct_type,think_process=think_process) # 保存agent对话
                if current_user_type and current_user_type.type_name == "exponential":

                    current_user_type.update_tokens(ntoken)
                sen1, answer_type,_,ntoken = formulate_gpt_question(None, result, class_type, "domain_answer",
                                                                    option) # 获取用户对领域问题的回答
                print("user:", sen1)
                if current_user_type and current_user_type.type_name == "exponential":
                    prospect=None
                    _,tolerance=current_user_type.should_end_conversation()
                    tolerance=int(tolerance * 1000) / 1000
                elif current_user_type and current_user_type.type_name == "expectation":
                    tolerance=None
                    _,prospect=current_user_type.should_end_conversation()
                    prospect=int(prospect * 1000) / 1000
                else:
                    prospect=None
                    tolerance=None
                save_conversation("user", sen1, answer_type,class_type,tolerance=tolerance,prospect=prospect) # 保存用户对话
    elif infe=='domain_confirm': # 领域确认
        while 1: # 循环直到确认领域
            a=0
            result1, answer_type,_,_ = formulate_gpt_question(None, sen, types, infe) # 调用GPT接口进行领域确认
            result1=full_convert(result1) # 全角转换
            result1 = ast.literal_eval(result1) # 将字符串转换为列表
            print(len(result1))
            if len(result1)>1: # 如果返回多个结果，需要进一步澄清
                if len(result1)==0:
                    result1=types # 如果结果为空，则使用候选类型列表
                result, answer_type,think_process,ntoken = formulate_gpt_question(None, sen, result1, 'domain_question') # 询问用户选择哪个领域
                usertoken = list(set(user_demand.key) - set(remain_demand.key))

                a=f'当前已进行{int(len(current_conversations)//2)}轮对话，得到{len(usertoken)}项用户需求:{usertoken},'
                think_process=a+think_process

                save_conversation("agent", result, answer_type,result1,think_process=think_process) # 保存agent对话

                if current_user_type and current_user_type.type_name == "exponential":

                    current_user_type.update_tokens(ntoken)
                sen, answer_type,_,_ = formulate_gpt_question(None, result, class_type, "domain_answer",
                                                              option) # 获取用户对领域问题的回答
                print("user:", sen)
                if current_user_type and current_user_type.type_name == "exponential":
                    prospect=None
                    _,tolerance=current_user_type.should_end_conversation()
                    tolerance=int(tolerance * 1000) / 1000
                elif current_user_type and current_user_type.type_name == "expectation":
                    tolerance=None
                    _,prospect=current_user_type.should_end_conversation()
                    prospect=int(prospect * 1000) / 1000
                else:
                    prospect=None
                    tolerance=None
                save_conversation("user", sen, answer_type,class_type,prospect=prospect,tolerance=tolerance) # 保存用户对话
            elif len(result1)==1: # 如果只返回一个结果，则进行确认
                for i in types: # 检查结果是否在候选类型列表中
                    z= re.sub(r"\s+", "", result1[0]) # 去除空格
                    if i==z:
                        option.append(z) # 添加到推断结果列表
                        a=1
                        break
                if a==1: # 确认成功则跳出循环
                    break

    """ # domain和type的澄清 clarify (领域和类型澄清 - 注释部分，原代码中的澄清逻辑)
    while len(option) > 1:
        confirm_question, question_type = formulate_gpt_question(None, option, sen, "domain_confirm")
        print("agent(confirm_question):", confirm_question)
        save_conversation("agent", confirm_question, question_type)
        domain_answer, answer_type = formulate_gpt_question(None, confirm_question, correct_type, "domain_answer",
                                                            option)
        print("user:", domain_answer)
        save_conversation("user", domain_answer, answer_type)
        result, _ = formulate_gpt_question(None, sen, types, "domain_infer")
        option = result
    domain_or_class_type = option[0]"""
    return option[0] # 返回推断出的领域或类型


def infer_domain_and_class(trigger_sen):
    """
    推断领域和子类型。

    Args:
        trigger_sen (str): 触发句。

    Returns:
        tuple: 包含领域 (domains_domain_type), 子类型 (domain_class_type), 领域数据路径 (path), 领域索引 (index) 的元组。
    """
    all_domains, all_types, path = read_domain() # 读取所有领域和类型信息
    domains = ','.join(all_domains) # 将领域列表转换为逗号分隔的字符串
    # 判断领域
    domain = None
    while domain not in all_domains or domain is None: # 循环直到推断出有效领域
        domain = infer_domain_type(trigger_sen, domains, "domain_infer",class_domain) # 调用领域推断函数
    domains_domain_type = domain # 推断出的领域
    print("domains_domain_type:", domains_domain_type)
    # 判断子类型
    index = all_domains.index(domain) # 获取领域索引
    domain_types = all_types[index] # 获取该领域的所有子类型
    if len(domain_types) != 1: # 如果子类型数量大于1，需要进一步确认
        types = ','.join(domain_types) # 将子类型列表转换为逗号分隔的字符串
        infer_type = None
        while infer_type not in domain_types or infer_type is None: # 循环直到确认子类型
            print('class_type',class_type)
            infer_type = infer_domain_type(trigger_sen, domain_types, "domain_confirm",class_type) # 调用领域确认函数进行子类型确认
        domain_class_type = infer_type # 确认的子类型
        print("domain_class_type:", domain_class_type)
    else: # 如果子类型只有一个，则直接使用
        domain_class_type = domain_types[0]
    return domains_domain_type, domain_class_type, path,index # 返回推断出的领域、子类型、数据路径和领域索引

def pre_determine_demand_from_profile():
    """
    通过分析用户画像预先确定部分用户需求。
    """
    global user_profiles, profile_list, qtoken_list, demands_domain, demands_type

    if not user_profiles or not profile_list or not qtoken_list:
        print("用户画像或token列表信息不足，无法进行预先确定。")
        return

    profile_info_for_prompt = []
    for token, profile_data in profile_list.items():
        if token in qtoken_list: # 确保token还在qtoken_list中，可能已经被处理掉了
            profile_info_for_prompt.append({
                "token": token,
                "annotation": qtoken_list[token].annotation,
                "profile_key": profile_data.profile_key,
                "profile_value": profile_data.profile_value,
                "profile_info": profile_data.profile_info
            })

    if not profile_info_for_prompt:
        print("没有可用于用户画像预先确定的token。")
        return

    prompt_input = {
        "user_profiles": user_profiles,
        "profile_list": profile_info_for_prompt,
        "domain": demands_domain,
        "demand_type": demands_type
    }

    prompt_str = json.dumps(prompt_input, ensure_ascii=False) # 将prompt数据转换为json字符串

    llm_response, response_type, think_process, ntoken = formulate_gpt_question(
        demands_type, prompt_str, None, "profile_based_predetermination"
    )


    try:
        determined_demands = json.loads(llm_response) # 尝试将LLM响应解析为JSON
        think_process = f"用户领域和类型已确认，现在我将利用用户画像{json.dumps(user_profiles, ensure_ascii=False)}信息和问题领域:{demands_domain}-{demands_type}，尝试预先确定部分用户需求。通过分析用户画像中的已知信息和特征关联，我预测到以下内容可能是用户会需要的{determined_demands}。这将减少后续对话轮次，提高效率。"

        if isinstance(determined_demands, dict): # 检查解析结果是否为字典
            save_conversation(
                "agent",
                "根据用户画像预先确定了以下需求。",
                "profile_based_determination",
                target_feature=list(determined_demands.keys()), # 将确定的token作为target_feature
                user_tokens=list(determined_demands.keys()), # 将确定的键值对作为feature_value
                user_values=list(determined_demands.values()),
                think_process=think_process # 保存思考过程
            )
            # 在这里可以添加进一步处理 determined_demands 的逻辑，例如更新 remain_demand 或 processing_list
            for token, value in determined_demands.items():
                if token in qtoken_list: # 确保token还在qtoken_list中
                    if token in remain_demand.token:
                        remain_demand.value[remain_demand.token.index(token)] = value # 更新remain_demand中的value
                    if token not in user_said_demand.token: # 如果用户之前没说过这个token
                        user_said_demand.token.append(token) # 添加到user_said_demand
                        user_said_demand.value.append(value) # 添加value到user_said_demand

                        remove_from_demand(remain_demand, token) # 从remain_demand中移除
                        if token in qtoken_list.keys(): # 如果token还在qtoken_list中
                            del qtoken_list[token] # 从qtoken_list中删除
        else:
            save_conversation(
                "agent",
                "用户画像预先确定需求分析完成，但LLM返回结果格式不正确。",
                "profile_based_determination_fail",
                think_process="LLM返回的不是有效的JSON字典" # 保存失败的思考过程
            )
            print("LLM response is not a valid dictionary.") # 打印错误信息
    except json.JSONDecodeError:
        save_conversation(
            "agent",
            "用户画像预先确定需求分析完成，但LLM返回结果无法解析为JSON。",
            "profile_based_determination_fail",
            think_process="LLM返回的JSON格式不正确" # 保存失败的思考过程
        )
        print("Could not decode LLM response as JSON.") # 打印JSON解码错误信息

# trigger以及进行准备操作
def judgeDomain_Type(trigger_sen, trigger_type, user_tokens, user_values):
    """
    根据触发句判断领域和类型，并进行初始化准备工作。

    Args:
        trigger_sen (str): 触发句。
        trigger_type (str): 触发类型。
        user_tokens (list): 用户在触发句中提及的token列表。
        user_values (list): 用户在触发句中提及的value列表。
    """
    global remain_demand, class_type, origin_qtoken_list, user_profiles, profile_list, demands_domain, demands_type


    print("\nuser:", trigger_sen)
    # 记录
    if current_user_type and current_user_type.type_name == "exponential":
        prospect=None
        _,tolerance=current_user_type.should_end_conversation()
        tolerance=int(tolerance * 1000) / 1000
    elif current_user_type and current_user_type.type_name == "expectation":
        tolerance=None
        _,prospect=current_user_type.should_end_conversation()
        prospect=int(prospect * 1000) / 1000
    else:
        prospect=None
        tolerance=None

    save_conversation("user", trigger_sen, "开始," + trigger_type, user_tokens, user_tokens, user_values,tolerance=tolerance,prospect=prospect) # 保存用户对话

    demands_domain, demands_type, path, index = infer_domain_and_class(trigger_sen) # 推断领域和子类型

    global webPages, qtoken_list, possible_pages
    domain_path = path[index] # 获取领域数据路径
    # 该领域的所有网页
    webPages = load_webpages(domain_path, demands_domain) # 加载领域网页
    possible_pages = copy.deepcopy(webPages) # 初始化待选网页列表为所有网页
    # 该领域子领域的一些信息
    token_averages, page_key2token, valid_tokens, TF, valid_page_numbers = cal(webPages, domain_path, demands_type) # 计算token平均长度、网页key到token的映射、有效token等信息
    # 该子领域的所有token
    qtoken_list = read_token(token_averages, domain_path, TF, valid_tokens, valid_page_numbers) # 读取token列表
    origin_qtoken_list = copy.deepcopy(qtoken_list) # 复制原始token列表
    with open("./data/path_to_web.json", "r", encoding="utf-8") as file:
        paths = json.load(file) # 加载网页路径配置
    user_profile_path = None
    for path in paths:
        if path["domain"] == demands_domain:
            user_profile_path = path["path"] # 获取用户画像数据路径

    with open("./data/" + user_profile_path + "/user_profile.json", "r", encoding="utf-8") as file:
        profiles = json.load(file) # 加载用户画像信息
    # 该页面的user profile和token的对应信息
    profile_dic = {} # 用户画像key到token的映射字典
    information = {} # token到用户画像信息的映射字典
    for profile in profiles:
        for key, value in profile["user_info"].items():
            # 从profile的关键字映射到token
            profile_dic[key] = profile["token"] # 构建用户画像key到token的映射
        # 从token映射到information字典
        # information是profile关键字到info的映射
        information[profile["token"]] = profile["information"] # 构建token到用户画像信息的映射

    # 读取用户的user profile并且建立与token和info的关系
    profile_list = {} # 用户画像token到用户画像信息的映射列表
    # user_profiles是profile关键字和值
    for pro_key, pro_value in user_profiles.items():
        profile_info = USE_USERPROFILE() # 创建用户画像信息对象
        profile_info.profile_key = pro_key # 设置用户画像key
        profile_info.profile_value = pro_value # 设置用户画像value
        profile_info.profile_info = information[profile_dic[pro_key]][pro_key] # 设置用户画像详细信息
        profile_list[profile_dic[pro_key]] = profile_info # 构建用户画像token到用户画像信息的映射

    pre_determine_demand_from_profile()
    # *****************************
    # 推理trigger中的feature-value
    infer_and_judge(trigger_sen, trigger_type, user_tokens, None) # 推理并判断触发句中的特征-值对
    # global features_prompt
    # features_prompt = generate_feature_prompt(webPages, qtoken_list)
    # *****************************


# 由于推理错误或者漏推而进行的重新推理
def re_infer(token, answer, process_info):
    """
    对token进行重新推理。

    Args:
        token (str): 需要重新推理的token。
        answer (str): 用户回答。
        process_info (PROCESSINGTOKEN): token的处理信息对象。
    """
    global qtoken_list, last_confirmed_demand
    # 重置一些属性
    process_info.unsuccessful = 0 # 重置不成功次数
    process_info.infer_value = None # 重置推断值
    while process_info.unsuccessful <= 4 and process_info.infer_value is None: # 循环直到推断成功或不成功次数超过限制

        # 让infer_value有值
        while process_info.infer_value is None: # 循环直到推断出值

            # 一定要先澄清是否为感兴趣的token
            if process_info.interested is not True: # 如果用户对token不感兴趣，进行澄清
                result = clarify_token(token, user_said_demand.token) # 澄清用户是否对token感兴趣
                if "消极" in result: # 如果用户表示不感兴趣
                    del processing_list[token] # 从处理列表中删除token
                    break
                else: # 如果用户表示感兴趣
                    process_info.interested = True # 设置为感兴趣

            process_info.unsuccessful += 1 # 增加不成功次数
            process_info.infer_value, process_info.token_answer_type = assist(token, process_info, answer,
                                                                              qtoken_list[token]) # 调用辅助函数进行推理
        # 重新推理出值
        if process_info.infer_value is not None: # 如果推断出值
            if process_info.token_answer_type == EXPLICIT: # 如果是显式回答
                process_info.unsuccessful = -1 # 设置为成功
                break
            # 进行澄清
            else: # 如果不是显式回答，需要澄清
                correct = clarify(token, process_info, answer) # 进行澄清
                # 澄清发现是错误的
                if not correct: # 如果澄清结果为错误
                    process_info.infer_value = None # 重置推断值
                    # unsuccessful + 1在后续的步骤中，这里不需要重复写
                # 澄清发现是正确的
                else: # 如果澄清结果为正确
                    # value被确定了
                    process_info.unsuccessful = -1 # 设置为成功
                    break


# 验证环节(未解决的需求再解决 默认正确的需求进行确认 )
def verify():
    """
    验证用户需求环节。
    """
    global class_type, processing_list, possible_pages, user_demand, \
        origin_qtoken_list, qtoken_list, user_said_demand, features_prompt

    # 列举已经完成的token
    init_verify_sen = "好的。您还觉得有其他方面需要改进吗？" # 初始化验证话术
    print("agent:以下是通过刚刚的对话我所了解的需求，请您核对以下的列表，如果有错误或者遗漏请及时告诉我。" + "\n")
    verify_list = {} # 待验证token列表
    print('processing_list',processing_list.items())
    """for token, value in processing_list.items(): # 遍历处理列表中的token
        if value.infer_value is not None: # 如果推断出值
            if value.value_type != "范围": # 如果不是范围类型
                print(str(token) + ":" + str(value.infer_value)) # 打印 token:value
            else: # 如果是范围类型
                for key in value.options:
                    #if value.options[key] == value.infer_value:
                    if key == value.infer_value:
                        print(str(token) + ":" + str(key)) # 打印 token:范围值 (显示范围对应的key)
                        break
            verify_list.update(token)  # 展示给用户看的列表 (添加到待验证列表)"""
    for con in current_conversations: # 遍历对话列表
        if con.feature_value.keys() is not None :# 如果对话包含token信息
            for i in con.feature_value.keys():
                if con.feature_value[i] is not None:
                    verify_list[i]=con.feature_value[i] # 添加到agent推断的token集合

    usertoken = list(set(user_demand.key) - set(remain_demand.key))
    think_process=f'当前已进行{int(len(current_conversations)//2)}轮对话，得到{len(usertoken)}项用户需求:{usertoken}, 已包含必选项。连续2次询问用户均未得到有效回复，考虑用户可能没有新的需求，所以将捕获到的需求交由用户确认。'
    save_conversation("agent","以下是通过刚刚的对话我所了解的需求，请您核对以下的列表，如果有错误或者遗漏请及时告诉我。" + "\n"+str(verify_list),'agent_demands_verify',think_process=think_process)
    if current_user_type and current_user_type.type_name == "exponential":
        token_count=0
        for char in think_process:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符
                token_count += 2
            elif char.isalpha():  # 英文字符
                token_count += 1.3
        current_user_type.update_tokens(token_count)
    # no_need_token, wrong_token_value, none_token_value+missed_token_value是user知道的
    # none_token_value是agent知道的

    no_need_token, wrong_token_value, none_token_value, missed_token_value = check_verify_list(user_said_demand,
                                                                                               verify_list,
                                                                                               user_demand) # 检查验证列表，获取不需要的token、错误value的token、未提及但agent知道的token、遗漏的token

    # 只要不是完全确定，当前就要被加回来
    for token, _ in processing_list.items(): # 遍历处理列表
        if processing_list[token].unsuccessful != -1: # 如果token未完全确定
            if token not in qtoken_list: # 如果token不在待询问列表中，重新添加
                qtoken_list[token] = copy.deepcopy(origin_qtoken_list[token]) # 从原始token列表中复制

    # verify_list的features_prompt(未处理前)
    features_prompt = generate_feature_prompt(webPages, qtoken_list, verify_list) # 生成特征prompt
    # verify_list中的token的feature_prompt
    verify_list_feature_prompt = {}
    if features_prompt is None:
        features_prompt = {}
    a=0
    # 第一步：处理verify_list中用户不需要的token
    if len(no_need_token) > 0: # 如果有用户不需要的token
        a=1
        current_user_type.unmatched_keys_count-=len(no_need_token)
        no_need_answer = requirement_validation(class_type, 'no_need', no_need_token, qtoken_list) # 生成用户不需要token的验证回答
        print("user:", no_need_answer)
        if current_user_type and current_user_type.type_name == "exponential":
            prospect=None
            _,tolerance=current_user_type.should_end_conversation()
            tolerance=int(tolerance * 1000) / 1000
        elif current_user_type and current_user_type.type_name == "expectation":
            tolerance=None
            _,prospect=current_user_type.should_end_conversation()
            prospect=int(prospect * 1000) / 1000
        else:
            prospect=None
            tolerance=None
        annotation=[]
        for c in no_need_token:
            annotation.append(qtoken_list[c].annotation)


        save_conversation("user", no_need_answer, "verify-多余的feature",tolerance=tolerance,prospect=prospect,annotation=annotation) # 保存用户对话 (验证-多余特征)
        infer_no_need_token = token_infer(no_need_answer, None, verify_list_feature_prompt) # 推理回答中的token

        # 检查合法性
        new_infer_token = []
        for n_token in infer_no_need_token: # 检查推理出的token是否在待验证列表中
            if n_token in verify_list_feature_prompt:
                new_infer_token.append(n_token) # 添加到新的推理token列表
        infer_no_need_token = new_infer_token # 更新推理token列表
        # 不进行clarify确认，直接删除
        for n_token in infer_no_need_token: # 遍历推理出的不需要的token
            if processing_list[n_token].interested is not True: # 如果用户对token不感兴趣
                del processing_list[n_token] # 从处理列表中删除token
        agent_sen, y,think_process ,ntoken= formulate_gpt_question(None, init_verify_sen, None, "ask_for_verify") # 生成询问验证的话术
        print("agent:", agent_sen)
        usertoken = list(set(user_demand.key) - set(remain_demand.key))
        a=f'当前已进行{int(len(current_conversations)//2)}轮对话，得到{len(usertoken)}项用户需求:{usertoken},'
        think_process=a+think_process
        save_conversation("agent", agent_sen, "verify-多余的feature",think_process=think_process) # 保存agent对话 (验证-多余特征)
        if current_user_type and current_user_type.type_name == "exponential":
            current_user_type.update_tokens(ntoken)
    # 第二步：处理verify list中错误的value（一次性说出来）
    if len(wrong_token_value) > 0: # 如果有错误value的token
        a=1
        current_user_type.unmatched_keys_count+=len(wrong_token_value)
        wrong_value_answer = requirement_validation(class_type, 'wrong_value', wrong_token_value,
                                                    qtoken_list) # 生成错误value的验证回答
        print("user:", wrong_value_answer)
        if current_user_type and current_user_type.type_name == "exponential":
            prospect=None
            _,tolerance=current_user_type.should_end_conversation()
            tolerance=int(tolerance * 1000) / 1000
        elif current_user_type and current_user_type.type_name == "expectation":
            tolerance=None
            _,prospect=current_user_type.should_end_conversation()
            prospect=int(prospect * 1000) / 1000
        else:
            prospect=None
            tolerance=None
        annotation=[]
        for c in wrong_token_value:
            if c in qtoken_list.keys():
                annotation.append(qtoken_list[c].annotation)

        save_conversation("user", wrong_value_answer, "verify-错误value",tolerance=tolerance,prospect=prospect,annotation=annotation) # 保存用户对话 (验证-错误value)
        infer_wrong_value_token = token_infer(wrong_value_answer, None, verify_list_feature_prompt) # 推理回答中的token

        new_infer_token = []
        # 合法性检验
        for token in infer_wrong_value_token: # 检查推理出的token是否在待验证列表中
            # 防止修改已经确定正确的token
            if token in verify_list_feature_prompt:
                new_infer_token.append(token) # 添加到新的推理token列表
        infer_wrong_value_token = new_infer_token # 更新推理token列表
        # infer_wrong_value_token：verify list中unsuccessful不为-1的
        for token in infer_wrong_value_token: # 遍历推理出的错误value的token
            if token in processing_list: # 如果token在处理列表中
                processing_info = processing_list[token] # 获取token处理信息
                processing_info.dialog_history.append(wrong_value_answer) # 添加用户回答到对话历史
                # 重新推理，需要correct
                # token肯定是被需要的
                re_infer(token, wrong_value_answer, processing_info) # 重新推理
            else: # 如果token不在处理列表中 (可能之前被误删了)
                # 可能第一步被误删了（clarify在后面，这里先无脑加进来）
                processing_token = PROCESSINGTOKEN() # 创建新的处理token对象
                processing_token.value_type = qtoken_list[token].token_type # 设置value类型
                processing_token.user_value = user_said_demand.value[user_said_demand.token.index(token)] # 设置用户value
                processing_token.dialog_history.append(wrong_value_answer) # 添加用户回答到对话历史
                processing_list[token] = processing_token # 添加到处理列表
                re_infer(token, wrong_value_answer, processing_token) # 重新推理
        agent_sen, o,think_process,ntoken = formulate_gpt_question(None, init_verify_sen, None, "ask_for_verify") # 生成询问验证的话术
        print("agent:", agent_sen)
        usertoken = list(set(user_demand.key) - set(remain_demand.key))
        a=f'当前已进行{int(len(current_conversations)//2)}轮对话，得到{len(usertoken)}项用户需求:{usertoken},'
        think_process=a+think_process
        save_conversation("agent", agent_sen, "verify-错误的value",think_process=think_process) # 保存agent对话 (验证-错误value)
        if current_user_type and current_user_type.type_name == "exponential":
            current_user_type.update_tokens(ntoken)
    # 第三步： 用户提到过的但是没在verify列表中看见
    if len(no_need_token) > 0 or len(missed_token_value) > 0: # 如果有不需要的token或遗漏的token
        a=1
        missed_all = none_token_value + missed_token_value # 合并未提及但agent知道的token和遗漏的token
        missed_answer = requirement_validation(class_type, 'missed_token', missed_all, qtoken_list) # 生成遗漏token的验证回答
        print("user:", missed_answer)
        if current_user_type and current_user_type.type_name == "exponential":
            prospect=None
            _,tolerance=current_user_type.should_end_conversation()
            tolerance=int(tolerance * 1000) / 1000
        elif current_user_type and current_user_type.type_name == "expectation":
            tolerance=None
            _,prospect=current_user_type.should_end_conversation()
            prospect=int(prospect * 1000) / 1000
        else:
            prospect=None
            tolerance=None
        annotation=[]
        for c in missed_all:
            annotation.append(qtoken_list[c].annotation)
        save_conversation("user", missed_answer, "verify-遗漏feature",tolerance=tolerance,prospect=prospect,annotation=annotation) # 保存用户对话 (验证-遗漏特征)

        prompt_list = copy.deepcopy(qtoken_list) # 复制待询问token列表
        for token in verify_list: # 移除已验证的token
            if token in prompt_list:
                del prompt_list[token] # 从prompt列表中删除已验证token
        # 这样推测出来的token就不会于verify list有重合了
        features_prompt = generate_feature_prompt(webPages, prompt_list) # 重新生成特征prompt (排除已验证token)
        # 推测token
        infer_missed_tokens = token_infer(missed_answer, none_token_value, features_prompt) # 推理回答中的token

        new_infer_token = []
        for token in infer_missed_tokens: # 检查推理出的token是否在prompt列表中
            if token in prompt_list:
                # 合法性检查
                new_infer_token.append(token) # 添加到新的推理token列表
        infer_missed_tokens = new_infer_token # 更新推理token列表

        for token in infer_missed_tokens: # 遍历推理出的遗漏token
            if token in none_token_value: # 如果token是未提及但agent知道的token
                processing_info = processing_list[token] # 获取token处理信息
                processing_info.dialog_history.append(missed_answer) # 添加用户回答到对话历史
                # 是之前推测过的token
                # token肯定是被需要的
                re_infer(token, missed_answer, processing_info) # 重新推理
            else: # 如果token是新的遗漏token
                # 没有推测过的token
                processing_token = PROCESSINGTOKEN() # 创建新的处理token对象
                processing_token.value_type = qtoken_list[token].token_type # 设置value类型
                processing_token.user_value = user_said_demand.value[user_said_demand.token.index(token)] # 设置用户value
                # 将回答存入对话历史中
                processing_token.dialog_history.append(missed_answer) # 添加用户回答到对话历史
                processing_list[token] = processing_token # 添加到处理列表
                # token不一定是被需要的
                re_infer(token, missed_answer, processing_token) # 重新推理

    record_infer_feature() # 记录推断出的特征
    # 重置
    processing_list = {} # 清空处理列表
    user_said_demand = DEMAND() # 清空用户提及的需求
    if a==0:
        print("user:没有错误了，谢谢")
        save_conversation('user', "没有错误了，谢谢",'endserve')



def record_infer_feature():
    """
    记录成功推断出的特征，并更新待询问token列表。
    """
    global processing_list
    # 逐个获取成功infer的kv的selected_qtoken并记录
    # processing_list中的token一定是用户提到过的(很小的概率是用户没提到但是模型推理出来value，并且value在answer中出现了)
    qtoken_to_remove = [] # 待移除的token列表
    for s_token in processing_list: # 遍历处理列表
        if processing_list[s_token].unsuccessful == -1: # 如果token推断成功
            last_confirmed_demand[s_token] = processing_list[s_token] # 添加到最后确认的需求列表
        if processing_list[s_token].interested: # 如果用户对token感兴趣
            # 用户已经回答对应的要求并且是有效的
            # 没有确定是正确的就被抛弃了
            if s_token in qtoken_list: # 如果token在待询问列表中
                qtoken_to_remove.append(s_token) # 添加到待移除列表
    for token in qtoken_to_remove: # 遍历待移除列表
        del qtoken_list[token] # 从待询问列表中删除token


# 模拟对话
def simulation_process():
    """
    模拟对话过程。
    """
    global qtoken_list, user_demand, remain_demand, current_user_type, interactions_rounds, MAX_DIALOGUE_ROUNDS # Add interactions_rounds and MAX_DIALOGUE_ROUNDS to global

    is_end = False # 对话是否结束标志
    # 必问的token
    required_tokens_union = set() # 存储所有网页的必问token
    for webpage in webPages: # 遍历网页列表
        required_tokens_union.update(webpage.required_tokens) # 合并必问token
    required_tokens = list(required_tokens_union) # 转换为列表
    while is_end is False and len(qtoken_list) != 0 and interactions_rounds // 2 < MAX_DIALOGUE_ROUNDS: # 当对话未结束且还有待询问token且未达到最大轮数时循环 # **添加轮数限制条件**
        # 检查用户类型是否需要结束对话
        aa,_=current_user_type.should_end_conversation()
        if current_user_type and aa :
            print("\n用户类型要求结束对话\n")
            is_end = True
            break

        # 在对话期间的verify,如果到的processing_list数量>=3就要verify
        if len(processing_list.keys()) >= 3: # 如果处理列表中的token数量达到3个，进行验证
            verify() # 进行验证
            # 验证后更新用户类型的匹配需求数
            if current_user_type and current_user_type.type_name == "expectation":
                matched_count, _, _, _, _ = evaluate_demand_matching(user_demand, current_conversations)
                current_user_type.update_needs(len(user_demand.token), matched_count)

        # 提问方式
        way = random.randint(0, 9) # 随机选择提问方式

        # 先提问当前网页中的required
        required_token = None
        remove_required = []

        # 第一轮问答：当用户trigger中没有需求的时候不要开放询问
        if way < 5 or len(list(qtoken_list.keys())) == len(list(origin_qtoken_list.keys())): # 根据提问方式或是否为第一轮问答选择提问策略
            is_end = one_token_process(required_token) # 一次询问一个token
        else:
            is_end = open_ended_process() # 开放式询问
    else: # **添加超出最大轮数时的结束逻辑**
        if interactions_rounds // 2 >= MAX_DIALOGUE_ROUNDS:
            print("\n达到最大对话轮数，结束对话。\n")
            is_end = True

    # end之后
    # end之后
    print("\n需求询问环节结束\n") # 打印需求询问环节结束提示
    verify() # 最后进行验证


# 开放式询问
def open_ended_process():
    """
    开放式询问过程。
    """
    global class_type, demands_domain, qtoken_list, unwanted_token_times, user_said_demand, profile_list, token_count, user_tokennum
    unwanted_token_times = 0 # 重置不感兴趣token计数器
    # 剩余的未问过的token
    max_len = len(qtoken_list) # 获取待询问token列表长度
    random_num = min(2, max_len) # 随机选择token数量，最多3个
    selectable_qtoken = {} # 可选择的token列表
    # 不要问在process list中的词
    for token, item in qtoken_list.items(): # 遍历待询问token列表
        if token not in processing_list: # 排除处理列表中的token
            selectable_qtoken[token] = item # 添加到可选择token列表


    # 全部在process list中，就verify,process list中数量已经够多了
    if len(selectable_qtoken) == 0: # 如果没有可选择的token，结束对话
        return True
    random_tokens=[]
    random_tokens = random.sample(list(selectable_qtoken), random_num) # 随机选择token
    random_annotations = [] # 随机token的annotation列表
    # 获取示例的token和annotation
    for token in random_tokens: # 遍历随机选择的token
        current_user_type.unmatched_keys_count-=1
        random_annotations.append(qtoken_list[token].annotation) # 获取annotation
    for token in random_tokens:
        qtoken_list.pop(token)
    # agent
    question, question_type, think_process,ntoken = formulate_gpt_question(demands_type, random_tokens, random_annotations,
                                                                           'open_ended_question') # 生成开放式问题
    print("开放式询问:")
    print("agent:", question)
    usertoken = list(set(user_demand.key) - set(remain_demand.key))
    a=f'当前已进行{int(len(current_conversations)//2)}轮对话，得到{len(usertoken)}项用户需求:{usertoken},'
    think_process=a+think_process
    save_conversation("agent", question, question_type, None, random_tokens,think_process=think_process) # 保存agent对话 (开放式询问)
    if current_user_type and current_user_type.type_name == "exponential":
        current_user_type.update_tokens(ntoken)
    # user
    answer, answer_type, user_tokens, user_values, user_kvs = user_choose_token(class_type, question, remain_demand,
                                                                                qtoken_list, profile_list)
    annotations=[]
    for token, value in zip(user_tokens, user_values): # 遍历用户回答的token和value
        if token not in user_said_demand.token: # 如果token不在用户提及的需求列表中
            user_said_demand.token.append(token) # 添加到用户提及的需求列表
            user_said_demand.value.append(value) # 添加value到用户提及的需求列表


            if token in qtoken_list.keys():
                annotations.append(qtoken_list[token].annotation)
                qtoken_list.pop(token)
            current_user_type.unmatched_keys_count+=1
            remove_from_demand(remain_demand, token) # 从剩余需求列表中移除token

    print("user:", answer)
    if current_user_type and current_user_type.type_name == "exponential":
        prospect=None
        _,tolerance=current_user_type.should_end_conversation()
        tolerance=int(tolerance * 1000) / 1000
    elif current_user_type and current_user_type.type_name == "expectation":
        tolerance=None
        _,prospect=current_user_type.should_end_conversation()
        prospect=int(prospect * 1000) / 1000
    else:
        prospect=None
        tolerance=None
    save_conversation("user", answer, answer_type, user_tokens, user_tokens, user_values,tolerance=tolerance,prospect=prospect,annotation=annotations) # 保存用户对话
    # 判断用户需求是否结束
    #demand_end, p, think_process,_ = formulate_gpt_question(None, question, answer, "is_end") # 判断用户需求是否结束
    #if "B" in demand_end: # 如果用户需求结束
    if len(user_demand.token) !=0:
        infer_and_judge(answer, answer_type, user_tokens, None) # 推理并判断回答中的特征-值对
        return False # 对话未结束，继续进行
    else: # 如果用户需求未结束
        return True # 对话结束


# 一次询问一个token
def one_token_process(chosen_token=None):
    """
    一次询问一个token的过程。

    Args:
        chosen_token (str): 选择询问的token，如果为None，则自动选择。

    Returns:
        bool: 对话是否结束。
    """
    global qtoken_list, current_conversations, unwanted_token_times, class_type, demands_domain, profile_list, \
        user_said_demand, features_prompt, demands_type, processing_list, token_count, current_user_type,umatch, user_tokennum
    if chosen_token is None: # 如果没有指定token
        # 选择要询问的token，select_token是询问策略
        q_token = select_token(qtoken_list, processing_list) # 选择要询问的token
    else: # 如果指定了token
        # 询问必备的token
        q_token = chosen_token # 使用指定的token
    # 所有token都被问过了或者没问过的也在processing list中了
    if q_token is None: # 如果没有可询问的token，结束对话
        return True
    selected_qtoken = qtoken_list[q_token]  # 获取qtoken对象
    annotation = selected_qtoken.annotation  # 获取annotation
    # 只是获取features_prompt中的例子，不用更新
    features_prompt = generate_feature_prompt(webPages, qtoken_list) # 生成特征prompt
    example_value = features_prompt[q_token][1] # 获取示例value
    print("当前询问的token为：", q_token)
    #qtoken_list.pop(q_token)
    # ***********************
    # GPT生成问句
    question, gpt_type,think_process ,ntoken= formulate_gpt_question(demands_type, q_token, annotation, "question", example_value) # 生成问题
    # question = "请问您对于房子的面积有什么特别的需求吗？"
    # gpt_type = "question"
    # ***********************
    usertoken = list(set(user_demand.key) - set(remain_demand.key))
    a=f'当前已进行{int(len(current_conversations)//2)}轮对话，得到{len(usertoken)}项用户需求:{usertoken},'
    think_process=a+think_process
    print("agent：", question)

    if current_user_type and current_user_type.type_name == "exponential":
        current_user_type.update_tokens(ntoken)
    # 用户回答
    # 不在需求列表里
    if q_token not in user_demand.token: # 如果询问的token不在用户需求列表中
        save_conversation("agent", question, str(gpt_type) + "询问", q_token,think_process=think_process) # 保存agent对话 (单token询问)

        answer, _ ,_,_= formulate_gpt_question(demands_type, question, None, 'no_interest') # 生成用户不感兴趣的回答
        print("user:", answer)
        unwanted_token_times = unwanted_token_times + 1 # 增加不感兴趣token计数器
        if current_user_type and current_user_type.type_name == "exponential":
            prospect=None
            _,tolerance=current_user_type.should_end_conversation()
            tolerance=int(tolerance * 1000) / 1000
        elif current_user_type and current_user_type.type_name == "expectation":
            tolerance=None
            current_user_type.unmatched_keys_count-=1
            _,prospect=current_user_type.should_end_conversation()
            prospect=int(prospect * 1000) / 1000
        else:
            prospect=None
            tolerance=None

        save_conversation("user", answer, "无兴趣回答",tolerance=tolerance,prospect=prospect) # 保存用户对话 (无兴趣回答)
        # 用户对此token无兴趣，不用再询问
        del qtoken_list[q_token] # 从待询问列表中删除token
        # 当多次没有选中用户想要的token时
        if unwanted_token_times == 2: # 如果不感兴趣token计数器达到2次，进行开放式询问
            is_end = open_ended_process() # 进行开放式询问
            return is_end # 返回对话是否结束

    # 若选择的token在需求列表中
    else: # 如果询问的token在用户需求列表中
        unwanted_token_times = 0 # 重置不感兴趣token计数器

        if q_token not in remain_demand.token:
            print(f"Warning: q_token '{q_token}' not found in remain_demand.token. Skipping processing.") # Optional warning
            return False
        else:
            save_conversation("agent", question, str(gpt_type) + "询问", q_token,think_process=think_process) # 保存agent对话 (单token询问)

        demand_index = remain_demand.token.index(q_token) # Now safe - q_token is guaranteed to be in remain_demand.token
        kvalue = remain_demand.value[demand_index] # 获取用户需求的value
        # 生成回答（可以带1-2个未问到的）
        answer, answer_type, user_tokens, user_values,annotations= expression_generation(class_type, question, q_token, kvalue,
                                                                                         remain_demand, qtoken_list,
                                                                                         profile_list) # 生成用户回答
        for token, value in zip(user_tokens, user_values): # 遍历用户回答的token和value
            if token not in user_said_demand.token: # 如果token不在用户提及的需求列表中
                user_said_demand.token.append(token) # 添加到用户提及的需求列表
                user_said_demand.value.append(value) # 添加value到用户提及的需求列表
                current_user_type.unmatched_keys_count+=1
                remove_from_demand(remain_demand, token) # 从剩余需求列表中移除token
                if token in qtoken_list.keys():
                    del qtoken_list[token]
            else:
                current_user_type.unmatched_keys_count-=1
        if current_user_type and current_user_type.type_name == "exponential":
            prospect=None
            _,tolerance=current_user_type.should_end_conversation()
            tolerance=int(tolerance * 1000) / 1000
        elif current_user_type and current_user_type.type_name == "expectation":
            tolerance=None
            _,prospect=current_user_type.should_end_conversation()
            prospect=int(prospect * 1000) / 1000
        else:
            prospect=None
            tolerance=None
        print("user:", answer)
        save_conversation("user", answer, answer_type + "回答", q_token, user_tokens, user_values,tolerance=tolerance,prospect=prospect,annotation=annotations) # 保存用户对话 (单token回答)
        # 推理
        infer_and_judge(answer, answer_type, user_tokens, [q_token]) # 推理并判断回答中的特征-值对
    return False # 对话未结束


# 对推理的token向用户进行确认
def clarify_token(infer_token, user_tokens):
    """
    澄清用户是否对推断出的token感兴趣。

    Args:
        infer_token (str): 推断出的token。
        user_tokens (list): 用户已提及的token列表。

    Returns:
        str: 用户的情感倾向，"积极" 或 "消极"。
    """
    # agent
    infer_token_qtoken = qtoken_list[infer_token] # 获取推断token的qtoken对象
    infer_token_annotation = infer_token_qtoken.annotation # 获取annotation
    usertoken = list(set(user_demand.key) - set(remain_demand.key))
    question, question_type,think_process,ntoken = formulate_gpt_question(None, infer_token, "clarify_token", infer_token_annotation) # 生成澄清问题
    a=f'当前已进行{int(len(current_conversations)//2)}轮对话，得到{len(usertoken)}项用户需求:{usertoken},'
    think_process=a+think_process
    save_conversation("agent", question, question_type, infer_token,infer_token,think_process=think_process) # 保存agent对话 (token澄清)
    if current_user_type and current_user_type.type_name == "exponential":
        current_user_type.update_tokens(ntoken)
    print("agent(token_clarify):", question)
    # user
    emotion, answer = infer_token_judgement(question, infer_token, user_tokens) # 用户判断token是否感兴趣
    if emotion == "积极":
        current_user_type.unmatched_keys_count+=1
    else:
        current_user_type.unmatched_keys_count-=1
    print("user:", answer)
    if current_user_type and current_user_type.type_name == "exponential":
        prospect=None
        _,tolerance=current_user_type.should_end_conversation()
        tolerance=int(tolerance * 1000) / 1000
    elif current_user_type and current_user_type.type_name == "expectation":
        tolerance=None
        _,prospect=current_user_type.should_end_conversation()
        prospect=int(prospect * 1000) / 1000
    else:
        prospect=None
        tolerance=None
    save_conversation("user", answer, None, infer_token,tolerance=tolerance,prospect=prospect) # 保存用户对话
    return emotion # 返回用户情感倾向


# 推理及后续处理
def infer_and_judge(answer, answer_type, user_tokens, certain_tokens):
    """
    推理用户回答中的token和value，并进行后续处理，包括澄清和辅助确认。

    Args:
        answer (str): 用户回答。
        answer_type (str): 回答类型。
        user_tokens (list): 用户回答中提及的token列表。
        certain_tokens (list, optional): 确定的token列表，用于辅助推理。 Defaults to None.
    """
    global qtoken_list, remain_demand, features_prompt, processing_list, user_demand, profile_list
    features_prompt = generate_feature_prompt(webPages, qtoken_list) # 生成特征prompt
    # 推断answer涉及的所有token
    # ***************
    valid = False
    new_infer_tokens = []
    new_infer_tokens = token_infer(answer, qtoken_list, features_prompt) # 推理回答中的token


    if certain_tokens is not None: # 如果有确定的token列表
        for token in certain_tokens: # 遍历确定token列表
            if token not in new_infer_tokens: # 如果确定token不在新推断的token列表中
                new_infer_tokens.append(token) # 添加到新推断的token列表
                if "None" in new_infer_tokens: # 移除 "None" token
                    new_infer_tokens.remove("None")

    # 回答语句中不包含需求（完全模糊触发句）
    if len(new_infer_tokens) == 0: # 如果没有推断出token，直接返回
        return
    infer_tokens = [token for token in new_infer_tokens if token in qtoken_list] # 过滤掉不在待询问列表中的token
    # 保存推断出来的token和value的信息，以及在文件中的信息
    for token in infer_tokens: # 遍历推断出的token
        # 最近一次用户回答可能涉及的feature
        processing_token = PROCESSINGTOKEN() # 创建新的处理token对象
        processing_token.value_type = qtoken_list[token].token_type # 设置value类型
        processing_token.infer_value, processing_token.options = feature_value_infer_llm(qtoken_list, webPages, token,
                                                                                         answer, profile_list) # 推理token的value和选项
        # 范围型不是存储具体的值，而是存储范围
        if processing_token.value_type == "范围" and processing_token.infer_value is not None: # 如果是范围类型且推断出值
            processing_token.infer_value =processing_token.infer_value # 设置推断值为范围
        # 是user在上一次回答中提到的value
        if token in user_said_demand.token: # 如果token在用户提及的需求列表中
            processing_token.user_value = user_said_demand.value[user_said_demand.token.index(token)] # 设置用户value
        processing_token.token_answer_type = answer_type # 设置回答类型
        # 将回答存入对话历史中"clarify_token"
        processing_token.dialog_history.append("user:" + answer) # 添加用户回答到对话历史
        processing_list[token] = processing_token # 添加到处理列表

    if certain_tokens is not None: # 如果有确定的token列表
        for token in certain_tokens: # 遍历确定token列表
            if token in processing_list.keys():
                processing_list[token].interested = True # 设置为感兴趣

    # 对于回答中的每一个token，要能够得到一个infer_value不管正误
    for infer_token in infer_tokens: # 遍历推断出的token
        process_info = processing_list[infer_token] # 获取token处理信息
        correct_token = None
        while process_info.unsuccessful < 4 and process_info.infer_value is None: # 循环直到推断成功或不成功次数超过限制

            # 用户在回答中明确提到了token，暂时认为是感兴趣的(但是不改变interest属性)
            if (infer_token in answer) or (process_info.interested is True): # 如果token在回答中提及或用户感兴趣
                correct_token = True # 认为是正确的token
            # 辅助和重述阶段
            # 为某一次回答中的每一个token配对一个value（value是网页中有的）
            # unsuccessful是推测出None的次数
            # 并不要求infer_value是正确的，只要有就行
            if (correct_token is not True) or (
                    process_info.unsuccessful == 1 and process_info.interested is not True) or (
                    process_info.value_type != "选项" and process_info.interested is not True): # 根据条件判断是否需要澄清
                # 第二个判断条件是为了纠正【回答中提到token就认为有兴趣】的潜在问题
                # 第三个判断条件是因为范围和数值的都要先确定有要求，因为assist会一直问直到问到正确结果
                # 只询问【当前这句话】是不是有需求
                result = clarify_token(infer_token, user_tokens) # 澄清用户是否对token感兴趣
                if "消极" in result: # 如果用户表示不感兴趣
                    # 只从需要待推测的token列表中删除
                    processing_list[infer_token].interested = False # 设置为不感兴趣
                    del processing_list[infer_token] # 从处理列表中删除token
                    break
                else: # 如果用户表示感兴趣
                    correct_token = True # 认为是正确的token
                    process_info.interested = True # 设置为感兴趣
            # 未确认value(infer_value不在可选value中，或根本就没推测出来)
            # infer不成功，需要辅助确认或者用户重述
            process_info.unsuccessful += 1 # 增加不成功次数
            # 如果这个值是因为存在于选项中被用户选择了，则大概率是正确的
            process_info.infer_value, process_info.token_answer_type = assist(infer_token, process_info, answer,
                                                                              qtoken_list[infer_token]) # 调用辅助函数进行推理

            # 可能是存在于选项中，也可能是用户重述了
            if process_info.infer_value is not None: # 如果推断出值
                process_info.interested = True # 设置为感兴趣
            if process_info.infer_value is not None and process_info.token_answer_type == EXPLICIT: # 如果是显式回答
                # 存在于选项中或者重述的时候type是EXPLICIT了
                process_info.unsuccessful = -1 # 设置为成功
            if process_info.infer_value is None and (
                    process_info.value_type == "数值" or process_info.value_type == "范围"): # 如果推断值为空且为数值或范围类型
                continue # 继续循环

    for s_token in processing_list: # 遍历处理列表
        # 将确定的用户在此次对话中说到的过token从待询问的列表中删除（保证之后不会再问到）,即使当前推出来的是None，后面verify时用户会说
        process_info = processing_list[s_token] # 获取token处理信息
        # 只是防止后面token infer时的干扰，所以对于 只是推出值但是没有完全肯定的不删除
        if process_info.interested and s_token in qtoken_list: # 如果用户感兴趣且token在待询问列表中
            del qtoken_list[s_token] # 从待询问列表中删除token


# agent辅助用户进行确认，带值询问
def assist(feature, process_info, requirement, qtoken):
    """
    agent辅助用户进行确认，根据token类型选择不同的辅助策略。

    Args:
        feature (str): 需要辅助确认的token。
        process_info (PROCESSINGTOKEN): token的处理信息对象。
        requirement (str): 用户回答。
        qtoken (QTOKEN): token的qtoken对象。

    Returns:
        tuple: 推断出的value和回答类型。
    """
    global class_type, demands_domain
    if qtoken.token_type == "选项": # 如果是选项类型
        # 随机选择 5 个值
        if len(process_info.options) > 5: # 如果选项数量超过5个，随机选择5个
            examples = random.sample(process_info.options, 5) # 随机选择5个选项
        else: # 否则使用所有选项
            examples = process_info.options # 使用所有选项
        # 去掉被问过的
        process_info.options = [example for example in process_info.options if example not in examples] # 从选项列表中移除已提问的选项
        return options_search(process_info, feature, requirement, examples) # 调用选项型辅助搜索函数
    elif qtoken.token_type == "数值": # 如果是数值类型
        return binary_search(process_info, feature, qtoken) # 调用数值型二分搜索函数
    elif qtoken.token_type == "范围": # 如果是范围类型
        return range_search(process_info, feature, qtoken, requirement) # 调用范围型搜索函数


# 选项型
def options_search(process_info, feature, requirement, examples, sorted_map=None):
    """
    选项型token的辅助搜索。

    Args:
        process_info (PROCESSINGTOKEN): token的处理信息对象。
        feature (str): token。
        requirement (str): 用户回答。
        examples (list): 选项列表。
        sorted_map (dict, optional): 排序后的选项映射，用于范围类型。 Defaults to None.

    Returns:
        tuple: 推断出的value和回答类型。
    """
    global class_type, demands_domain, qtoken_list
    matching_qtoken = qtoken_list[feature] # 获取qtoken对象
    annotation = matching_qtoken.annotation # 获取annotation
    # agent
    # *******
    retell_probability = random.randint(0, 1)  # 询问语句中可能需要用户重述 (随机决定是否需要用户重述)
    # retell_probability = 1
    # *******
    if retell_probability == 0 or process_info.value_type == "范围": # 如果不需要重述或为范围类型

        question, gpt_type,think_process,ntoken = formulate_gpt_question(demands_domain, feature, requirement, "assist_options",
                                                                         examples,
                                                                         annotation) # 生成选项型辅助问题
    else: # 如果需要重述
        question, gpt_type,think_process,ntoken = formulate_gpt_question(demands_domain, feature, requirement, "assist_and_retell",
                                                                         examples,
                                                                         annotation) # 生成选项型辅助和重述问题
    print("agent(assist):", question)
    usertoken = list(set(user_demand.key) - set(remain_demand.key))
    a=f'当前已进行{int(len(current_conversations)//2)}轮对话，得到{len(usertoken)}项用户需求:{usertoken},'
    think_process=a+think_process
    process_info.dialog_history.append("agent:" + str(question)) # 添加agent问题到对话历史
    save_conversation("agent", question, gpt_type, feature,examples,think_process=think_process) # 保存agent对话 (选项型辅助)
    if current_user_type and current_user_type.type_name == "exponential":
        current_user_type.update_tokens(ntoken)
    association_value = None # 关联value
    a_type = None # 回答类型
    if feature in user_said_demand.token: # 如果token在用户提及的需求列表中
        user_value = user_said_demand.value[user_said_demand.token.index(feature)] # 获取用户value
    else: # 如果token不在用户提及的需求列表中
        user_value = None # 用户value为空
    if process_info.value_type == "选项": # 如果是选项类型
        if str(user_value) in question and user_value is not None: # 如果用户value在问题中且不为空
            # 存在于选项中
            answer, answer_type = requirement_confirmation(class_type, question, user_value, "options_exist") # 生成选项存在确认回答
            print("user:", answer)
            current_user_type.unmatched_keys_count+=1
            if current_user_type and current_user_type.type_name == "exponential":
                prospect=None
                _,tolerance=current_user_type.should_end_conversation()
                tolerance=int(tolerance * 1000) / 1000
            elif current_user_type and current_user_type.type_name == "expectation":
                tolerance=None
                _,prospect=current_user_type.should_end_conversation()
                prospect=int(prospect * 1000) / 1000
            else:
                prospect=None
                tolerance=None
            save_conversation("user", answer, answer_type,user_value,tolerance=tolerance,prospect=prospect) # 保存用户对话 (选项存在确认)
            association_value, _ = feature_value_infer_llm(qtoken_list, webPages, feature, answer) # 推理回答中的value
            a_type = EXPLICIT # 回答类型为显式
        else: # 如果用户value不在问题中或为空
            # 不存在于选项中，但是是用户感兴趣的
            if feature in user_said_demand.token: # 如果token在用户提及的需求列表中
                # 不存在于选项中，则有一定几率重述，可能重述两到三次
                if retell_probability == 1: # 如果需要重述
                    # 重述后会再一次 be inferred
                    association_value, a_type = retell(feature, user_value, process_info) # 调用重述函数
            if feature not in user_said_demand.token or retell_probability != 1: # 如果token不在用户提及的需求列表或不需要重述
                # 不重述
                # （用户不感兴趣 or 正确选项不存在）
                current_user_type.unmatched_keys_count-=1
                answer, answer_type = requirement_confirmation(class_type, question, user_value, "options_not_exist") # 生成选项不存在确认回答
                print("user:", answer)
                process_info.dialog_history.append("user:" + str(answer)) # 添加用户回答到对话历史
                if current_user_type and current_user_type.type_name == "exponential":
                    prospect=None
                    _,tolerance=current_user_type.should_end_conversation()
                    tolerance=int(tolerance * 1000) / 1000
                elif current_user_type and current_user_type.type_name == "expectation":
                    tolerance=None
                    _,prospect=current_user_type.should_end_conversation()
                    prospect=int(prospect * 1000) / 1000
                else:
                    prospect=None
                    tolerance=None
                save_conversation("user", answer, answer_type,tolerance=tolerance,prospect=prospect) # 保存用户对话 (选项不存在确认)
    # 范围型的单位可能不一样,所以返回的是sorted_map中的范围值
    elif process_info.value_type == "范围": # 如果是范围类型
        user_range = sorted_map[user_value] # 获取用户value对应的范围
        # user
        association_value = None # 关联value
        a_type = None # 回答类型
        for value in examples: # 遍历选项列表
            if sorted_map[value] == user_range: # 如果选项对应的范围与用户范围相同
                current_user_type.unmatched_keys_count+=1
                answer, answer_type = requirement_confirmation(class_type, question, user_value, "options_exist") # 生成选项存在确认回答
                print("user:", answer)
                if current_user_type and current_user_type.type_name == "exponential":
                    prospect=None
                    _,tolerance=current_user_type.should_end_conversation()
                    tolerance=int(tolerance * 1000) / 1000
                elif current_user_type and current_user_type.type_name == "expectation":
                    tolerance=None
                    _,prospect=current_user_type.should_end_conversation()
                    prospect=int(prospect * 1000) / 1000
                else:
                    prospect=None
                    tolerance=None
                save_conversation("user", answer, answer_type,user_value,tolerance=tolerance,prospect=prospect) # 保存用户对话 (选项存在确认)
                association_value, _ = feature_value_infer_llm(qtoken_list, webPages, feature, answer) # 推理回答中的value
                association_value = sorted_map[association_value] # 获取排序后的关联value
                a_type = EXPLICIT # 回答类型为显式
                break
        if a_type != EXPLICIT: # 如果回答类型不是显式
            answer, answer_type = requirement_confirmation(class_type, question, user_value, "options_not_exist") # 生成选项不存在确认回答
            print("user:", answer)
            current_user_type.unmatched_keys_count-=1
            if current_user_type and current_user_type.type_name == "exponential":
                prospect=None
                _,tolerance=current_user_type.should_end_conversation()
                tolerance=int(tolerance * 1000) / 1000
            elif current_user_type and current_user_type.type_name == "expectation":
                tolerance=None
                _,prospect=current_user_type.should_end_conversation()
                prospect=int(prospect * 1000) / 1000
            else:
                prospect=None
                tolerance=None
            save_conversation("user", answer, answer_type,tolerance=tolerance,prospect=prospect) # 保存用户对话 (选项不存在确认)
    return association_value, a_type # 返回关联value和回答类型


def range_search(process_info, feature, qtoken, requirement=None):
    """
    范围型token的搜索。

    Args:
        process_info (PROCESSINGTOKEN): token的处理信息对象。
        feature (str): token。
        qtoken (QTOKEN): token的qtoken对象。
        requirement (str, optional): 用户回答。 Defaults to None.

    Returns:
        tuple: 推断出的value和回答类型。
    """
    sorted_map = copy.deepcopy(process_info.options) # 复制选项列表
    sorted_map = {i + 1: value for i, value in enumerate(sorted_map)} # 构建选项序号到选项值的映射


    sorted_token = [] # 排序后的token序号列表
    sorted_value = [] # 排序后的value列表
    # 保存排序之后的options并去重
    for key, value in sorted_map.items(): # 遍历排序后的选项映射
        if value  in sorted_value: # 如果value已在排序后的value列表中 (去重逻辑，实际代码中可能存在问题，因为此处判断条件永远为False，value不可能在sorted_value中)
            sorted_token.append(key) # 添加到排序后的token序号列表
            sorted_value.append(value) # 添加到排序后的value列表

    while len(sorted_token) != 0: # 循环直到找到范围或排序列表为空
        # 取中值
        index = (len(sorted_token) - 1) // 2 # 计算中间索引
        # 具体的值
        middle_value = sorted_token[index] # 获取中间序号
        # agent
        # 生成问句
        question, gpt_type,think_process,ntoken = formulate_gpt_question(demands_type, feature, process_info.dialog_history,
                                                                         "assist_numeric", middle_value,qtoken ) # 生成数值型辅助问题 (范围类型也使用数值型辅助问题模板)
        process_info.dialog_history.append("agent：" + question) # 添加agent问题到对话历史
        print("agent(assist numeric):", question)
        usertoken = list(set(user_demand.key) - set(remain_demand.key))
        a=f'当前已进行{int(len(current_conversations)//2)}轮对话，得到{len(usertoken)}项用户需求:{usertoken},'
        think_process=a+think_process
        save_conversation("agent", question, gpt_type, feature,middle_value,think_process=think_process) # 保存agent对话 (数值型辅助 - 范围类型)
        if current_user_type and current_user_type.type_name == "exponential":
            current_user_type.update_tokens(ntoken)
        # user
        value = process_info.user_value # 获取用户value
        # 判断大小并且给出回复

        answer, answer_type = requirement_confirmation(class_type, question, value, "range",
                                                       middle_value, sorted_map) # 生成范围型确认回答

        process_info.dialog_history.append("user：" + answer) # 添加用户回答到对话历史
        print("user:", answer)
        if current_user_type and current_user_type.type_name == "exponential":
            prospect=None
            _,tolerance=current_user_type.should_end_conversation()
            tolerance=int(tolerance * 1000) / 1000
        elif current_user_type and current_user_type.type_name == "expectation":
            tolerance=None
            _,prospect=current_user_type.should_end_conversation()
            prospect=int(prospect * 1000) / 1000
        else:
            prospect=None
            tolerance=None
        save_conversation("user", answer, answer_type,feature,value,tolerance=tolerance,prospect=prospect) # 保存用户对话 (范围型确认)
        # 判断user的回答蕴含的大小关系

        compare_result, _,_ ,_= formulate_gpt_question(None, middle_value, answer, "range_result") # 判断用户回答的范围比较结果
        if "A" in compare_result: # 如果用户回答表示小于中间值
            example_to_remove = [] # 待移除的选项序号列表
            for i in range(index, len(sorted_token)): # 移除大于等于中间值的选项序号
                example_to_remove.append(sorted_token[i])
            for example in example_to_remove: # 遍历待移除列表
                sorted_token.remove(example) # 移除选项序号
        elif "B" in compare_result: # 如果用户回答表示等于中间值
            # 有风险，可能是回答的风险
            process_info.infer_value = process_info.options[middle_value] # 设置推断值为中间值对应的范围
            break # 结束循环
        elif "C" in compare_result: # 如果用户回答表示大于中间值
            example_to_remove = [] # 待移除的选项序号列表
            for i in range(0, index + 1): # 移除小于等于中间值的选项序号
                example_to_remove.append(sorted_token[i])
            for example in example_to_remove: # 遍历待移除列表
                sorted_token.remove(example) # 移除选项序号
        elif "D" in compare_result: # 如果用户回答表示与中间值范围有交集
            # 即有交集，将有交集的元素放在一起去做option assist
            middle_range = sorted_map[middle_value] # 获取中间值对应的范围
            approximate_range = [] # 近似范围的选项序号列表
            tokens = sorted_token # 获取当前选项序号列表
            for key in tokens: # 遍历选项序号列表
                key_range = sorted_map[key] # 获取选项序号对应的范围
                if middle_range[1] <= key_range[0] or middle_range[0] >= key_range[1]: # 如果中间值范围与选项范围没有交集，跳过
                    continue
                else: # 如果有交集，添加到近似范围列表
                    approximate_range.append(key)
            if len(approximate_range) > 1: # 如果近似范围列表长度大于1，使用选项型辅助搜索
                process_info.infer_value, a_type = options_search(process_info, feature, requirement, approximate_range,
                                                                  process_info.options) # 调用选项型辅助搜索
            else: # 如果近似范围列表长度不大于1，推断值为中间值对应的范围
                process_info.infer_value = process_info.options[middle_value] # 设置推断值为中间值对应的范围
            break # 结束循环
    infer_value = None # 推断值
    a_type = None # 回答类型
    # B选项或D选项
    if process_info.infer_value is not None: # 如果推断出值
        infer_value = process_info.infer_value # 设置推断值
        a_type = EXPLICIT # 回答类型为显式
    # 可能某一次判断错误了
    return infer_value, a_type # 返回推断值和回答类型


def binary_search(process_info, feature, qtoken):
    """
    数值型token的二分搜索。

    Args:
        process_info (PROCESSINGTOKEN): token的处理信息对象。
        feature (str): token。
        qtoken (QTOKEN): token的qtoken对象。

    Returns:
        tuple: 推断出的value和回答类型。
    """
    results, _,_,_ = formulate_gpt_question(None, process_info.options, None, "sort_numerical")  # 将数值型的value排序 (调用GPT接口对数值型选项排序)
    start_index = results.find("['") # 查找排序结果的起始索引
    end_index = results.find("']") # 查找排序结果的结束索引
    sorted_examples = results[start_index + 2:end_index].split("', '") # 解析排序后的选项列表
    not_sorted_options = [x for x in process_info.options if x not in sorted_examples] # 获取未排序的选项列表 (排序接口可能返回部分选项，此处获取未排序的剩余选项)
    # 4-5次即可
    # 循环
    while True: # 循环进行二分搜索
        # 取中值
        index = (len(sorted_examples) - 1) // 2 # 计算中间索引
        if index>len(sorted_examples): # 修复索引越界问题，当sorted_examples为空时，index可能为-1，导致越界
            middle_value = sorted_examples[0] # 取第一个元素作为中间值 (当sorted_examples为空时，此处仍然会报错，需要进一步处理空列表情况，但根据代码逻辑，sorted_examples不应该为空)
        else: # 索引未越界
            middle_value = sorted_examples[index] # 获取中间值
        if len(sorted_examples) == 1: # 如果排序后的选项列表只剩一个元素，推断值为该元素
            process_info.infer_value = middle_value # 设置推断值为该元素
            break # 结束循环
        # agent
        # 生成问句
        question, gpt_type,think_process,ntoken = formulate_gpt_question(demands_type, feature, process_info.dialog_history,
                                                                         "assist_numeric", middle_value, qtoken) # 生成数值型辅助问题
        process_info.dialog_history.append("agent：" + question) # 添加agent问题到对话历史
        print("agent(assist numeric):", question)
        usertoken = list(set(user_demand.key) - set(remain_demand.key))
        a=f'当前已进行{int(len(current_conversations)//2)}轮对话，得到{len(usertoken)}项用户需求:{usertoken},'
        think_process=a+think_process
        save_conversation("agent", question, gpt_type, feature,middle_value,think_process=think_process) # 保存agent对话 (数值型辅助)
        if current_user_type and current_user_type.type_name == "exponential":
            current_user_type.update_tokens(ntoken)
        # user
        value = process_info.user_value # 获取用户value
        # 判断大小并且给出回复

        answer, answer_type = requirement_confirmation(class_type, question, value, "numeric",
                                                       middle_value) # 生成数值型确认回答

        process_info.dialog_history.append("user：" + str(answer)) # 添加用户回答到对话历史
        print("user:", answer)
        if current_user_type and current_user_type.type_name == "exponential":
            prospect=None
            _,tolerance=current_user_type.should_end_conversation()
            tolerance=int(tolerance * 1000) / 1000
        elif current_user_type and current_user_type.type_name == "expectation":
            tolerance=None
            _,prospect=current_user_type.should_end_conversation()
            prospect=int(prospect * 1000) / 1000
        else:
            prospect=None
            tolerance=None
        save_conversation("user", answer, answer_type,feature,value,tolerance=tolerance,prospect=prospect) # 保存用户对话 (数值型确认)
        # 判断user的回答蕴含的大小关系

        compare_result, _ ,_,_= formulate_gpt_question(None, question, answer, "numeric_result") # 判断用户回答的数值比较结果

        if "A" in compare_result: # 如果用户回答表示小于中间值
            example_to_remove = [] # 待移除的选项列表
            for i in range(index, len(sorted_examples)): # 移除大于等于中间值的选项
                example_to_remove.append(sorted_examples[i])
            for example in example_to_remove: # 遍历待移除列表
                sorted_examples.remove(example) # 移除选项
        elif "B" in compare_result: # 如果用户回答表示等于中间值
            # 有风险
            process_info.infer_value = middle_value # 设置推断值为中间值
            break # 结束循环
        elif "C" in compare_result: # 如果用户回答表示大于中间值
            example_to_remove = [] # 待移除的选项列表
            for i in range(0, index + 1): # 移除小于等于中间值的选项
                example_to_remove.append(sorted_examples[i])
            for example in example_to_remove: # 遍历待移除列表
                sorted_examples.remove(example) # 移除选项
    infer_value = None # 推断值
    a_type = None # 回答类型
    # B选项
    if process_info.infer_value is not None: # 如果推断出值
        a_type = EXPLICIT # 回答类型为显式
    process_info.options = not_sorted_options + sorted_examples # 更新选项列表 (将未排序的选项和排序后的剩余选项合并)
    return infer_value, a_type # 返回推断值和回答类型


# 重述
def retell(token, value, process_info):
    """
    用户重述需求。

    Args:
        token (str): token。
        value (str): 用户value。
        process_info (PROCESSINGTOKEN): token的处理信息对象。

    Returns:
        tuple: 重述后的推断值和回答类型。
    """
    global class_type, profile_list
    # user
    retell_answer, answer_type = retell_generator(class_type, process_info, token, value,
                                                  qtoken_list,
                                                  profile_list) # 生成重述回答
    print("user:", retell_answer)
    process_info.dialog_history.append("user:" +str(retell_answer)) # 添加用户回答到对话历史
    if current_user_type and current_user_type.type_name == "exponential":
        prospect=None
        _,tolerance=current_user_type.should_end_conversation()

        tolerance=int(tolerance * 1000) / 1000
    elif current_user_type and current_user_type.type_name == "expectation":
        tolerance=None
        _,prospect=current_user_type.should_end_conversation()
        prospect=int(prospect * 1000) / 1000
    else:
        prospect=None
        tolerance=None
    current_user_type.unmatched_keys_count+=1
    save_conversation("user", retell_answer, str(answer_type) + "回答", token,tolerance=tolerance, prospect=prospect) # 保存用户对话 (重述回答)
    infer_value, _ = feature_value_infer_llm(qtoken_list, webPages, token, retell_answer) # 推理重述回答中的value
    return infer_value, answer_type # 返回推断值和回答类型


# agent向用户询问以澄清
def clarify(feature, process_info, sentence):
    """
    agent向用户澄清value。

    Args:
        feature (str): 需要澄清的token。
        process_info (PROCESSINGTOKEN): token的处理信息对象。
        sentence (str): 用户原始回答。

    Returns:
        bool: 澄清结果，True表示用户确认，False表示用户否认。
    """
    # 该语句是澄清语句,澄清用户回答的所有token
    global current_conversations, class_type, demands_domain
    # agent
    user_value = process_info.user_value # 获取用户value
    if process_info.value_type != "范围": # 如果不是范围类型
        infer_value = process_info.infer_value # 获取推断值
    else: # 如果是范围类型
        for option in process_info.options: # 遍历选项列表
            if process_info.options[option] == process_info.infer_value: # 查找推断值对应的选项
                infer_value = option # 设置推断值为选项序号
                break

    clarify_question, sen_type,think_process,ntoken = formulate_gpt_question(demands_type, feature, infer_value, "clarify", sentence) # 生成澄清问题
    print("agent(clarify)：", clarify_question)
    usertoken = list(set(user_demand.key) - set(remain_demand.key))
    a=f'当前已进行{int(len(current_conversations)//2)}轮对话，得到{len(usertoken)}项用户需求:{usertoken},'
    think_process=a+think_process
    process_info.dialog_history.append("agent:" + str(clarify_question)) # 添加agent问题到对话历史
    save_conversation("agent", clarify_question, sen_type, feature, None, infer_value,think_process=think_process) # 保存agent对话 (澄清)
    if current_user_type and current_user_type.type_name == "exponential":
        current_user_type.update_tokens(ntoken)
    # user

    if process_info.value_type != "范围": # 如果不是范围类型
        user_value = process_info.user_value # 获取用户value
    else: # 如果是范围类型
        user_value = process_info.options[process_info.user_value] # 获取用户value对应的范围值
    infer_value = process_info.infer_value # 获取推断值
    current_user_type.unmatched_keys_count+=1
    result, answer = requirement_judgment(class_type, user_value, infer_value, clarify_question) # 用户判断澄清问题
    print("user:", answer)
    process_info.dialog_history.append("user:" +str(answer)) # 添加用户回答到对话历史
    if current_user_type and current_user_type.type_name == "exponential":
        prospect=None
        _,tolerance=current_user_type.should_end_conversation()
        tolerance=int(tolerance * 1000) / 1000
    elif current_user_type and current_user_type.type_name == "expectation":
        tolerance=None
        _,prospect=current_user_type.should_end_conversation()
        prospect=int(prospect * 1000) / 1000
    else:
        prospect=None
        tolerance=None
    save_conversation("user", answer, None, infer_value,tolerance=tolerance,prospect=prospect) # 保存用户对话
    return result # 返回澄清结果 (True/False)

def evaluate_demand_matching(user_demand, current_conversations):
    """
    评估需求匹配度，计算完整率和冗余率。
    Args:
        user_demand (DEMAND): 用户原始需求对象。
        current_conversations (list): 当前对话列表。
    Returns:
        tuple: 包含匹配的用户需求数量、agent推断的需求token列表、用户原始需求列表、完整率、冗余率的元组。
    """
    matched_count = 0  # 匹配的需求数量
    unmatched_count = 0  # 未匹配的需求数量
    user_demands_list = []  # 用户原始需求列表
    agent_tokens_from_conversation = set()  # 从对话中提取的 agent 推断的 token 集合

    # 提取 agent 提到的 tokens
    for con in current_conversations:  # 遍历对话列表
        if con.feature_value.keys() is not None:  # 如果对话包含 token 信息
            for i in con.feature_value.keys():  # 遍历 token 信息
                agent_tokens_from_conversation.add(i)  # 添加到 agent 推断的 token 集合
        if con.target_feature is not None and con.role == 'agent':  # 如果对话是 agent 提出的且包含目标特征
            if isinstance(con.target_feature, list):  # 如果目标特征是列表
                agent_tokens_from_conversation.update(con.target_feature)  # 更新 agent 推断的 token 集合
            elif isinstance(con.target_feature, str):  # 如果目标特征是字符串
                agent_tokens_from_conversation.add(con.target_feature)  # 添加到 agent 推断的 token 集合

    # 构建用户原始需求字典
    user_demands_dict = {user_demand.token[i]: user_demand.value[i] for i in range(len(user_demand.token))}

    # 计算匹配数量和未匹配数量
    for agent_token in agent_tokens_from_conversation:  # 遍历 agent 推断的 token 集合
        if agent_token in user_demands_dict:  # 如果 agent 提问的 token 存在于用户需求中
            matched_count += 1  # 匹配数量加 1
        else:  # 如果 agent 提问的 token 不在用户需求中
            unmatched_count += 1  # 未匹配数量加 1

    # 更新用户的期望需求和匹配需求
    expected_needs = len(user_demands_dict)  # 用户的期望需求总数
    matched_needs = matched_count  # 匹配的需求数量

    # 计算完整率和冗余率
    completeness_rate = round((matched_needs / expected_needs) * 100 if expected_needs else 0, 1)
    redundancy_rate = round((unmatched_count / max(1, len(agent_tokens_from_conversation))) * 100, 1)

    return matched_needs, list(agent_tokens_from_conversation), user_demands_list, completeness_rate, redundancy_rate



# 将对话写入文件
def write_to_file(all_demand_conversations):
    """
    将所有需求的对话信息写入JSON文件。

    Args:
        all_demand_conversations (list): 包含所有需求对话信息的列表。
    """

    with open("./data/conversations.json", "w", encoding="utf-8") as f: # 打开文件用于写入
        json.dump(all_demand_conversations, f, indent=4, ensure_ascii=False, ) # 将对话信息写入JSON文件，格式化缩进为4，不转义ASCII字符


def run():
    """
    主运行函数，执行整个对话流程并评估。
    """
    start_time = time.time() # 记录开始时间
    with open("./data/demands.json", "r", encoding="utf-8") as file: # 加载需求列表
        demands = json.load(file) # 从JSON文件加载需求
    num_demands = len(demands) # 获取需求数量
    all_demand_conversations = [] #存储所有需求的对话列表
    a=1
    start=0# 设置起始需求索引
    global token_count
    last_token=200
    # 设置用户类型
    selected_type = "normal"
    global current_user_type
    current_user_type = UserType(selected_type)
    print(f"\n选择用户类型: {selected_type}")

    for i, demand in tqdm(enumerate(demands), total=num_demands, desc="Processing Demands"): # 遍历需求列表，显示进度条
        current_user_type.unmatched_keys_count=0
        current_user_type.total_tokens=0
        while True:
            token_long=random.randint(50,200)
            if token_long-last_token>50 or token_long-last_token<50:
                last_token=token_long

                break
        current_user_type.token_long=token_long
        if a<start: # 跳过指定索引之前的需求
            a+=1
            continue

        demand_start_time = time.time() # 记录当前需求开始时间
        token_count = 0
        global current_conversations, interactions_rounds, user_demand, remain_demand, user_profiles, profile_list, final_pagesAkv, user_said_demand, last_confirmed_demand, processing_list, class_domain, class_type, demands_domain, demands_type, qtoken_list, origin_qtoken_list, webPages, possible_pages, features_prompt, unwanted_token_times, last_confirmed_demand
        current_conversations = [] #重置当前对话列表
        interactions_rounds = 0 # 重置交互轮次计数器
        user_demand = DEMAND() # 重置用户需求对象
        remain_demand = DEMAND() # 重置剩余需求对象
        user_profiles = {} # 重置用户画像信息
        profile_list = {} # 重置用户画像映射列表
        final_pagesAkv = {} # 重置最终网页-特征值对
        user_said_demand = DEMAND() # 重置用户提及的需求对象
        last_confirmed_demand = {} # 重置最后确认的需求列表
        processing_list = {} # 重置处理列表
        class_domain = None # 重置真实领域
        class_type = None # 重置真实类型
        demands_domain = None # 重置推断领域
        demands_type = None # 重置推断类型
        qtoken_list = {} # 重置待询问token列表
        origin_qtoken_list = {} # 重置原始token列表
        webPages = [] # 重置网页列表
        possible_pages = [] # 重置待选网页列表
        features_prompt = {} # 重置特征prompt
        unwanted_token_times = 0 # 重置不感兴趣token计数器

        # 判断用户需求的领域，生成trigger
        readDemand_and_judgeDomain(demand) # 读取需求并判断领域类型
        # 开始模拟
        simulation_process() # 开始模拟对话过程
        # 写入文件
        # 将token-value变成key-value并与网页页面对应
        for token in last_confirmed_demand: # 遍历最后确认的需求列表
            if token not in origin_qtoken_list: # 如果token不在原始token列表中，跳过
                continue
            selected_qtoken = origin_qtoken_list[token] # 获取qtoken对象
            save_qualify_value(final_pagesAkv, selected_qtoken, possible_pages, last_confirmed_demand[token]) # 保存合格的value和网页页面对应关系

        matched_count, agent_demands, user_demands, completeness_rate, redundancy_rate = evaluate_demand_matching(user_demand, current_conversations)
        demand_conversation_data = { # 存储当前需求的对话数据
            "demand_index": i+1, # 需求索引
            "conversations": [con.__dict__ for con in current_conversations], #对话列表
            "interaction_rounds": interactions_rounds // 2, # 交互轮次
            "matched_demand_count": matched_count, # 匹配的需求数量
            "completeness_rate": completeness_rate, # 完整率
            "redundancy_rate": redundancy_rate, # 冗余率
            "agent_demands": agent_demands, # agent推断的需求列表
            "original_user_need":{ # 用户原始需求
                "keys": user_demand.key, # 需求key列表
                "token-values": {token:value for token, value in zip(user_demand.token,user_demand.value)}, # 需求token列表
            },
            "user_type": selected_type # 添加用户类型信息
        }
        all_demand_conversations.append(demand_conversation_data) # 将当前需求的对话数据添加到总对话列表

        # 进度显示和时间估计
        elapsed_time_demand = time.time() - demand_start_time # 计算当前需求处理时间
        average_time_per_demand = (time.time() - start_time) / (i + 1) # 计算平均每个需求处理时间
        remaining_demands = num_demands - (i + 1) # 计算剩余需求数量
        estimated_remaining_time = remaining_demands * average_time_per_demand # 估计剩余处理时间
        print('estimated_remaining_time:', estimated_remaining_time)
        print(f"\nDemand {i+1}/{num_demands} processed.") # 打印需求处理完成信息
        print("\n用户原始需求:") # 打印用户原始需求
        for j in range(0, len(user_demand.key)): # 遍历用户原始需求key列表
            print(user_demand.key[j], ":", user_demand.value[j]) # 打印 key:value
        print("\nAgent对话得到的需求:") # 打印agent对话得到的需求
        for agent_token in agent_demands: # 遍历agent推断的需求token列表
            print(f"{agent_token}") # 打印 agent token
        print("交互轮次：", interactions_rounds // 2) # 打印交互轮次
        print(f"完整率: {completeness_rate:.2f}%") # 打印完整率
        print(f"冗余率: {redundancy_rate:.2f}%") # 打印冗余率
        print("conversation done") # 打印对话完成提示
        write_to_file(all_demand_conversations) # 将所有对话信息写入文件


if __name__ == '__main__':
    run()
