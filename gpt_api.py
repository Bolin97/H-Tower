import json
import random
import math
from openai import OpenAI
import requests

import default_data
from prompt import select_not_exist, select_exist, judge, token_infer, domain_infer, is_end_prompt, token_clarify, \
    sort_numbers, wrong_token, re_vague_trigger, ab_vague_trigger, explicit_answer, ab_vague_answer, re_vague_answer, \
    user_profile_answer, sort_range, missed_token,domaintype_infer,domaintype_question,token_question

EXPLICIT = default_data.EXPLICIT
RE_VAGUE_WITH_PROFILE = default_data.RE_VAGUE_WITH_PROFILE
BOUNDED = default_data.BOUNDED
RE_VAGUE_SYNONYM = default_data.RE_VAGUE_SYNONYM
AB_VAGUE = default_data.AB_VAGUE

VAGUE = default_data.VAGUE
RE_VAGUE = default_data.RE_VAGUE


def generate_history(param_2):
    result = ""
    for sen in param_2:
        result = "\n" + result + sen
    return result


def generate_demand(param_1, param_2, extra_1):
    result = ""
    if param_2 is not None:
        for i in range(0, len(param_1)):
            result = result + str(param_1[i]) + "(" + str(extra_1[i]) + "):" + str(param_2[i]) + "\n"
    else:
        for i in range(0, len(param_1)):
            result = result + str(param_1[i]) + "(" + str(extra_1[i]) + "),"
    return result

def generate_demands(param_1, param_2):
    result = ""
    if param_2 is not None:
        for i in range(0, len(param_1)):
            result = result + str(param_1[i]) +":" + str(param_2[i]) + "\n"
    else:
        for i in range(0, len(param_1)):
            result = result + str(param_1[i]) + ","
    return result

# TODO 改成agent user functional各一个函数，
# TODO 改成langchain清晰一些
def formulate_gpt_question(demand, param_1, param_2, s_type, extra_1=None, extra_2=None):
    question = "你好"
    gpt_type = EXPLICIT
    think_process = None

    # agent
    # 如果类型是问句
    if s_type == "question":
        result = ",".join(extra_1)
        question = "你是一名客服，你正在与用户聊天，你们已经聊了一会了。用户想要" + demand + "，你应该怎么向他询问关于\"" + param_1 + "\"的需求信息，" + \
                   "\"" + param_1 + "\"指\"" +  str(param_2) + "\"，例如\"" + result + "\"等等。请构造一个面向客户的询问语句，询问语句之后不需要提供可选项，" + \
                   "你的回答应该只有询问语句,询问语句要符合中文口语习惯，尽量简洁。\n"
        gpt_type = 'question'
        think_process = f"结合之前的对话,用户还没有提到{param_1}这个特征,我发现这个特征是当前领域中重要的参数，需要优先了解。考虑到用户的总体需求'{demand}'，我应该构造一个符合中文口语习惯的询问语句，不提供选项，让用户自由表达他们的需求"

    # 辅助确认
    elif s_type == "assist_options":
        result = ",".join(extra_1)
        question = "你是客服，你正在客户与对话,帮助他" + demand + "客户对于\"" + param_1 + "\"的要求是\"" +  str(param_2) + \
                   "\"。\n(注释:" + param_1 + "是指" + extra_2 + ")\n你认为用户的具体要求不清晰，" + \
                   "所以你想在下一次询问中给出客户几个可供选择的选项，选项为\"" + result + \
                   "\"。请构造下一次询问的问句。"
        gpt_type = "选项型辅助确认"
        think_process = f"我发现用户对{param_1}的表达'{param_2}'不够清晰，无法准确理解用户的具体需求。这种情况下，我应该提供一些选项帮助用户做选择，而不是继续开放式提问。我从当前领域的常见值中选择了几个可能匹配用户需求的选项：{result},提供选项的方式比继续开放式提问更有效,我需要构造一个包含这些选项的询问语句。"

    elif s_type == "assist_numeric":
        annotation = extra_2.annotation
        result = "\n".join( str(param_2))
        question = "你是一名客服，你正在与用户对话,帮助他" + demand + "，你正在确定用户对于\"" + param_1 + "\"的需求。" + \
                   "(" + param_1 + ":" + annotation + ")" + \
                   "以下是你们之前的聊天记录：\n" + result + "\n你认为用户的具体要求不清晰，所以你想向他询问他对于\"" + param_1 + \
                   "\"的需求是高于、低于还是等于" + str(extra_1) + "请接着构造一个问句" + str(extra_1) + \
                   "要包含在问句中，问话不要和之前的问话方式相同，多样性一些。\n"
        gpt_type = "数值型辅助确认"
        think_process = f"我正在确定用户对{param_1}的需求,从之前的对话记录中，我发现用户的表达不够清晰，无法确定具体的数值需求。对于数值型特征，使用比较法是高效的确认方式。我选择了一个参考值{extra_1}，这个值通常是通过对所有可能值排序后取中间位置得到的。我将询问用户的需求是高于、低于还是等于这个值，这样可以快速缩小范围。我需要确保问句中包含{extra_1}这个具体值"

    # 辅助确认+重述
    elif s_type == "assist_and_retell":
        result = ",".join(extra_1)
        question = "你是客服，你正在客户与对话，用户想要" + demand + "客户对于\"" + param_1 + "\"的要求是\"" +  str(param_2) + \
                   "\"。\n(注释:" + param_1 + "是指" + extra_2 + ")\n你认为用户的具体要求不清晰，现在你需要在询问中给出客户几个可供选择的选项，" + \
                   "选项为\"" + result + "并告知用户若选项都不是他的需求就请他再次详细描述需求。"
        gpt_type = "辅助确认+重述"
        think_process = f"我发现用户对{param_1}的表达'{param_2}'不够清晰，但与其仅提供选项，我决定同时给用户重新描述需求的机会,我从当前领域的常见值中选择了几个可能匹配用户需求的选项：{result},同时会明确告知用户，如果提供的选项都不符合需求，可以重新描述，"

    # value澄清
    elif s_type == "clarify":
        question = "你是一名客服，正在与用户对话,用户想要" + demand + "。用户回答是\"" + extra_1 + "\",你想要询问用户，关于他回答中的\"" + param_1 + \
                   "\"需求,他是否想表达他希望\"" + param_1 + "\"为\"" +  str(param_2) + "\"。请生成询问语句。"
        gpt_type = "澄清"
        think_process = f"我从用户的回答'{extra_1}'中发现了一些线索，可能表明用户对{param_1}有特定需求，具体值可能是{param_2}。这个推断是基于用户回答中的关键词和表达方式，但我不能完全确定，所以需要向用户确认。澄清比假设更安全，可以避免后续推荐不符合用户真实需求的选项。我需要构造一个询问语句，直接询问用户是否希望{param_1}为{param_2}。"

    # 开放式询问
    elif s_type == "open_ended_question":
        ex = ",".join(param_1)
        an = generate_demand(param_1, None, param_2)
        question = "你是一名客服,用户目的:" + demand + "。你想知道用户对什么方面还有需求，例如:" + ex + "\n(注释" + an + ")," + \
                   token_question
        gpt_type = "token_question"
        think_process = f"之前与用户的对话基本上已经确定了用户需求,现在还需要使用开放式询问来了解用户是否有更多需求。我需要从用户目的领域中的特征参数中未被询问的特征中选择几个可能的需求参数：{ex},来向用户询问是否有更多需求"

    # token澄清
    elif s_type == "clarify_token":
        question = token_clarify + "\n feature:" +  str(param_2) + "\n feature annotation:" + extra_1 + "\n you:"
        gpt_type = "clarify_token"
        think_process = f"我从用户之前的回答中发现了一些线索，可能表明用户对{param_2}这个特征有需求，但我不确定用户是否真的关心这个特征，所以需要确认用户是否关心这个特征本身。这种澄清可以避免询问用户不关心的特征，提高对话效率"

    elif s_type=='extract_feature_values':
        question=f"""你是一名需求分析大师,请分析以下对话，提取用户回答中关于特定特征的值：

        Agent问题: {param_1}
        用户回答: {param_2}
        
        需要提取的特征列表: {str(demand)},其对应的特征解释为{extra_1}
        范围类别的value,其中的到替换为-,示例:'30到50万字',应该输出为'30-50万字',
        仔细分析得到每个特征对应的值,不要遗漏特征
        请以JSON格式返回每个特征对应的值，格式为 {{"特征1": "值1", "特征2": "值2", ...}}。
        严格根据特征列表来对应生成,不要新建或修改特征名称,级别一般是指的是车的级别,例如:跑车
        """
        gpt_type = 'extract_feature_values'


    elif s_type == "profile_based_predetermination":
        prompt_data = json.loads(param_1) # 将json字符串解析为字典
        user_profiles = prompt_data["user_profiles"]
        profile_list = prompt_data["profile_list"]
        domain = prompt_data["domain"]
        demand_type = prompt_data["demand_type"]

        profile_list_str = json.dumps(profile_list, ensure_ascii=False) # 将profile_list转换为json字符串,方便prompt使用

        question = f"""你是一名智能客服助手，你的任务是根据用户画像（user profile）和已知的用户画像特征列表（profile_list），
        从待询问的token列表中，提前预测并确定那些可以基于用户画像直接确定的token的需求值。

        用户当前的需求领域是：{domain}
        用户需求的类型是：{demand_type}

        以下是用户的用户画像信息（user profile）：
        {json.dumps(user_profiles, ensure_ascii=False)}

        以下是用户画像特征列表（profile_list），包含了token，token解释，用户画像key，用户画像value以及用户画像详细信息：
        {profile_list_str}

        请分析用户画像和用户画像特征列表，判断哪些token的需求值可以基于用户画像直接确定。
        请返回一个JSON格式的字典，key是你可以确定的token，value是对应token的值。
        如果你无法确定任何token的值，或者没有任何token可以基于用户画像预先确定，请返回一个空的JSON字典 {{}}.

        请只返回JSON字典，不要返回任何其他文字。
        """
        gpt_type = "profile_based_predetermination"
    # user
    # 回答
    elif s_type == "answer":
        random_number = random.randint(0, 10)
        ques = param_1
        keys = extra_1
        values = extra_2
        if random_number <=8:  # 完全清楚
            re = generate_demands(keys, values)
            question = "你是一名用户,正在与客服对话，你们在之前已经聊过一段时间了，接下来你需要根据目的和需求列表" + \
                       "(**重要：回答请严格限制在需求列表中的需求，不要描述需求列表之外的需求**)，针对客服的问题进行回答。回答不要太正式" + \
                       explicit_answer + demand + "\n客服问题:" + ques + "需求列表:\n" + re
            gpt_type = EXPLICIT
        elif random_number == 9:  # 数值换个范围说，其他的用同义词替换
            re = generate_demands(keys, values)
            question = "你是用户，正在与客服对话，你们在之前已经聊过一段时间了，接下来你先对需求列表中的键值进行一些联想 " + \
                       "将需求列表中具体的值改写成描述性的话或者同义词。然后再回答客服的问题，回答时只需描述需求列表中的需求即可，不要额外添加需求" + \
                       "**重要：请确保你的回答只包含需求列表中的信息，不要引入任何新的需求或信息。** 回答不要太正式 " + re_vague_answer + demand + \
                       "\n客服问题:" + ques + "需求列表:\n" + re
            gpt_type = RE_VAGUE
        else:  # 完全模糊
            re = generate_demands(keys, values)
            question = "你是用户，正在与客服对话，你们在之前已经聊过一段时间了，接下来你先对需求列表中的键值结合自身情况进行一些联想" + \
                       "然后将需求列表中具体的值改写成模糊的描述性的语句。然后再回答客服的问题，回答时只用描述需求列表中的需求，不要额外添加需求。" + \
                       "回答不要太正式 ,改写后具体的值不应该出现在回答中。" + \
                       ab_vague_answer + demand + \
                       "\n客服问题:" + ques + "需求列表:\n" + re
            gpt_type = AB_VAGUE

    elif s_type == "add_user_profile":
        re = param_1
        que = param_2
        user_profiles = extra_1
        tokens = extra_2
        profile = []
        for token in tokens:
            if token in user_profiles:
                profile.append(user_profiles[token].profile_key)

        profiles = "\n".join(profile)
        question = "context：input中涉及多个需求，rewrite keys为input中的某些需求，user profile为已知的信息。" + \
                   "并且user profile与rewrite keys相对应。" + \
                   "\ninstruction：根据给出的user profile重写input中的rewrite keys，使其不再被具体的值形容而是用user profile形容,回答为output" + \
                   "注意：只要重写rewrite keys中的需求，句子的其他需求不变；重写的需求的具体值不应该出现在output中；改写后语句要符合中文口语习惯。回答内容不要回答与用户上输入无关的内容" + \
                   user_profile_answer + que + "\nrewrite keys:\n" + re + "user profile:\n" + profiles + "\noutput:\n"
        gpt_type = RE_VAGUE
    elif s_type == "end":
        question = "你是顾客，你正在与客服对话，你想" + demand + "。客服问你\"" + param_1 + "\"，请告诉客服你已经没有别的需求了，只要输出回答语句，" + \
                   "回答不要太正式，需求描述应该符合口语习惯，不要有多余的表达。"
        gpt_type = "需求结束"
    # 触发句
    elif s_type == "trigger":
        random_number = random.randint(0, 4)  #
        # 精准
        if 0 <= random_number <3:
            re = generate_demand(param_1, param_2, extra_1)
            question = "你是顾客，你正在与客服对话，请生成一句话作为开场白，表示你想要" + demand + "，你的需求有:\n" + re + \
                       "注意：开场白不要太正式 。"
            gpt_type = EXPLICIT

        # 同义词替换
        elif  random_number == 3:
            re = generate_demand(param_1, param_2, extra_1)
            question = "你是顾客，你正在与客服对话，请生成一句话作为开场白。首先你对需求列表中的键值进行一些联想 " \
                       "然将需求列表中具体的值改写成描述性的话或者同义词后再生成开场白。开场白语气不要太正式，" + \
                       + re_vague_trigger + demand + "\n需求:" + re

            gpt_type = RE_VAGUE_SYNONYM

        # 完全模糊
        elif  random_number ==4:
            re = generate_demand(param_1, param_2, extra_1)
            question = "你是顾客，你正在与客服对话，请生成一句话作为开场白。首先你对需求列表中的键值进行一些联想" \
                       "，将需求列表中具体的值改写成模糊的描述性的话。然后再生成开场白。开场白语气不要太正式，" + \
                       + ab_vague_trigger + demand + "\n需求:" + re
            gpt_type = AB_VAGUE

        # 不带需求
        else:
            question = "你是顾客，你正在与客服对话，请生成一句话作为开场白，表示你想要" + demand + "开场白不要太正式应该符合口语习惯"
            gpt_type = "不含需求"
    elif s_type == "no_interest":
        question = "请直接回复'都可以',不要回复其他内容"
        gpt_type = "no interest"
    # 需求判断句
    elif s_type == "judge":
        question = judge + param_1 + "\"\nagent infer:" + extra_1 + "\n你的需求:" +  str(param_2)
        gpt_type = "需求判断"
        think_process = f"我需要判断我对用户需求的理解是否准确。我的问题是'{param_1}'，我推断用户的需求是'{extra_1}'，而用户的实际需求是'{param_2}'。这种判断有助于确保我正确理解用户的需求，避免后续推荐不符合需求的选项。如果判断结果是匹配的，我可以继续基于这个理解进行对话；如果不匹配，我需要调整我的理解或提供更准确的选项。"
    # assist
    # 需求辅助确认
    elif s_type == "select":
        if extra_1 == "selection":
            question = select_exist + "客服:\"" + param_1 + "\"\n用户需求:" +  str(param_2)
            gpt_type = "存在于选项中"
        elif extra_1 == "not_exist":
            question = select_not_exist + "客服:\"" + param_1
            gpt_type = "不存在于选项中"
    elif s_type == "user_retell":
        history = "\n".join(extra_1)
        question = "你是一名用户，你正在与客户对话，你们正在沟通" + param_1 + "(" +  str(param_2) + ")方面的需求" + \
                   "以下是对话记录:\n<history>" + history + "</history>" + "\ninstruction:接下来你需要先表明agent提供的选项中没有你想要的，" + \
                   "然后改写以下需求描述,并给客服说。\n" + "需求描述:" + extra_2 + "\n注意需求描述不能和你之前的描述重复。\n"
    elif s_type == "lower_number":
        question = "你是一名用户,你正在与客服聊天，请告诉客服你的期望低于他的建议\n" + \
                   "不要在回答中出现具体的值,不要扩展回答。" \
                   + \
                   "\n客服:" + param_1
        gpt_type = "lower_number"
    elif s_type == "higher_number":
        question = "你是一名用户,你正在与客服聊天，请告诉客服你的期望高于他的建议\n" + \
                   "不要在回答中出现具体的值,不要扩展回答。" \
                   + \
                   "\n客服:" + param_1
        gpt_type = "higher_number"
    elif s_type == "right_number":
        re = "数值" if extra_1 == "numeric" else "范围"
        question = "你是一名用户,你正在与客服聊天，请告诉客服你的期望等于他的建议" + re + "的。给出你的回答" + \
                   "\n客服:" + param_1
        gpt_type = "right_number"
    elif s_type == "approximate_number":
        question = "你是一名用户你正在与客服聊天，请告诉客服你的期望与他的建议有重叠，但是不完全相同。给出你的回答" + \
                   "\n##" + \
                   "\n客服：请问您期望的价格是大于，等于或者小于300-500万呢" + \
                   "\n用户:嗯，我期待的价格与300到500万有重叠的地方，但是也不完全是" + \
                   "\n##" + \
                   "\n客服：" + param_1
        gpt_type = "approximate_number"
    # verify
    elif s_type == "ask_for_verify":
        question = "你是一名客服,请对用户的要求表示赞同，并询问他是否还认为有可以改进的地方,回答简洁直白\n"
        think_process = f"我已经收集了多个用户需求，现在是时候进行一次验证，确认我理解的用户需求是否准确完整。在验证过程中，我需要检查已处理的特征是否与用户的实际需求匹配，包括是否有遗漏的特征、错误理解的特征或错误的值,这个验证步骤可以避免对话结束时发现重大误解，提高整体对话效率和用户满意度"
    elif s_type == "no_need":
        results = ""
        for token in param_1:
            if token in param_2:
                results += str(token) + "(" + param_2[token].annotation + ")\n"
            else:
                results += str(token) + "(该特征为错误推断信息,请让客服丢弃)\n"
        question = "你是用户,你想要" + demand + "并且正在和客服聊天。\n" + "客服对你的需求理解出错了，请你生成语句告诉客服你对:\n" + results + \
                   "这些方面都没有要求。(回答语句不要提到其他的需求)\n"
    elif s_type == "wrong_value":
        results = ""
        for token in param_1:
            #           results += str(token) + "(" + param_2[token].annotation + ")\n"
            results += str(token) +"\n"
        question = "你是一名用户,正在与客服聊天。请你根据#错误内容#生成回答，告诉客服他的列表中的信息有误,回答简洁直白" + \
                   wrong_token + str(demand) + "\n错误内容:\n" + results
    elif s_type == "missed_token":
        results = ""
        for token in param_1:
            results += str(token) + "(" + param_2[token].annotation + ")\n"
        question = "你是一名用户,正在与客服聊天。请你根据#遗漏内容#生成回答，告诉客服他的列表中的信息有遗漏（不需要详细描述需求的具体内容）,回答简洁直白" + \
                   missed_token + str(demand) + "\n遗漏内容:\n" + results
    # 功能性
    # elif s_type is None:
    #     question = param_1 + "\n如果以上的句子表示没有特别的要求则输出消极，否则输出积极。"
    # elif s_type == 'emotion':
    #     question = "问题:" + param_1 + "\n回答：" + param_2 + "\n如果回答是对问题的否定就输出消极，否则输出积极。\n输出："
    #     gpt_type = "emotion"
    # 推测token
    elif s_type == "token_infer":
        question = token_infer + str(extra_1)  + "\nsentence:" + str(param_1)
        s_type = "token_infer"
        think_process = f"我需要从用户的句子'{param_1}'中推断出可能的需求token。我有一系列可能的token及其含义：{extra_1}，需要分析用户的表达，找出其中隐含的需求参数。这种推断可以帮助理解用户模糊表达中的具体需求，特别是当用户使用非标准术语或描述性语言时。准确的token推断可以减少后续澄清的需要，提高对话效率。"
    # 判断对话是否结束
    elif s_type == "is_end":
        #question = is_end_prompt + param_1 + "\n answer:" + param_2 + "\noptions:\nA:用户没有额外的需求了\nB:用户还有需求" + "\nresult:"
        answer_param = param_2 if param_2 is not None else ""
        question = is_end_prompt + str(param_1) + "\n answer:" + str(answer_param) + "\noptions:\nA:用户没有额外的需求了\nB:用户还有需求" + "\nresult:"
        think_process = f"我需要判断用户是否还有其他需求。根据我的问题'{param_1}'和用户的回答'{answer_param}'，我需要分析用户的回答是表示已经没有其他需求了，还是暗示还有未表达的需求。这种判断有助于决定是否继续对话或结束需求收集阶段。如果用户表示没有其他需求，我可以总结已收集的需求并结束对话；如果用户还有需求，我需要继续询问以捕获所有重要需求。"
    # 判断用户需求属于哪个领域和子类
    elif s_type == "domain_infer":
        question = domain_infer + str(param_1)
        gpt_type = "domain_infer"
        think_process = f"我需要从用户的表述'{param_1}'中推断出用户需求属于哪个领域和子类,作为后面对话的基础"
    elif s_type == "domain_confirm":
        question = domaintype_infer +  "##\n用户:" + str(param_1) + "\noptions:" + str(param_2)
        gpt_type = "domain_confirm"
        think_process = f"因为无法从用户语句中得到准确的领域类型,我需要确认用户的需求'{param_1}'是否属于我推断的领域类型, 需要为用户提供可能的领域选项：{param_2}，这种确认可以避免错误的领域分类，确保后续对话使用正确的知识库。如果我的推断正确，可以直接进入该领域的具体需求收集；如果不正确，需要调整领域分类或提供更多选项。"
    elif s_type == "domain_question":
        question = domaintype_question +  "##\n用户:" + str(param_1) + "\noptions:" + str(param_2)
        gpt_type = "domaintype_question"
        think_process = f"根据之前的对话内容,我现在需要生成一个问题，询问用户的需求'{param_1}'具体属于哪个领域子类。需要为用户提供可能的领域选项：{param_2}，但需要用户进一步确认。这种询问可以让用户主动选择最符合其需求的领域类型，提高后续对话的针对性。与直接推断相比，这种方法可以减少错误分类的可能性，特别是当用户的表述可能属于多个领域时。"
    elif s_type == "domain_answer":
        options = ",".join(extra_1)
        question = "你是用户,客服的问题\"" + str(param_1) + "\"\n 客服问题涉及的选项:" + str(options) + "\n 你的答案:" + str(param_2) + "\n 请根据客服的问题生成回答，" + \
                   "回答不要太正式，要符合口语习惯。"
    # assist
    elif s_type == "sort_numerical":
        numbers = ",".join(param_1)
        question = sort_numbers + numbers + "\n列表结果:"
        gpt_type = "sort_numerical"
        think_process = f"我需要对数值列表{numbers}进行排序。这是系统内部的辅助功能，用于准备数值型token的二分查找过程。排序后的结果将用于确定询问用户的中间值，帮助快速缩小用户需求的范围。"
    elif s_type == "sort_range":
        numbers = ",".join(param_1)
        question = sort_range + numbers + "\n列表结果:"
        gpt_type = "sort_range"
        think_process = f"我需要对范围列表{numbers}进行排序。这是系统内部的辅助功能，用于准备范围型token的查找过程。排序后的结果将用于确定询问用户的参考范围，帮助快速缩小用户需求的范围区间。"
    elif s_type == "numeric_result":
        question = "客服:" + str(param_1) + "\n用户:" +  str(param_2) + "\n以上对话中：\nA：用户的需求低于客服给出的范围\n" + \
                   "B：用户的需求等于客服给出的范围\nC：用户的需求高于客服给出的范围\n请给出正确选项"
        gpt_type = "numeric_result"
        think_process = f"我需要分析客服的问题'{param_1}'和用户的回答'{param_2}'，判断用户的需求是低于、等于还是高于客服给出的范围。这种判断有助于系统理解用户的数值型需求，进一步缩小搜索范围，最终确定用户的具体需求值。"
    elif s_type == "range_result":
        question = "客服:" + str(param_1) + "\n用户:" +  str(param_2) + "\n以上对话中：\nA：用户的需求低于客服给出的范围\n" + \
                   "B：用户的需求刚好等于客服给出的范围\nC：用户的需求高于客服给出的范围\nD:用户的需求与客服给出的范围有重合，但不完全相同\n请给出正确选项"
        gpt_type = "range_result"
        think_process = f"我需要分析客服的问题'{param_1}'和用户的回答'{param_2}'，判断用户的需求范围与客服给出的范围的关系。这种判断比数值型更复杂，需要考虑范围的重叠情况，有助于系统理解用户的范围型需求，最终确定用户的具体需求范围。"
    # 生成同意或者不同意的句子
    elif s_type == "agree":
        question = "你是用户,请给出肯定回答，不要扩充回答，回答中不要提到其他需求。" + "\n\nquestion:" + param_1 + "\nanswer:"

    elif s_type == "disagree":
        question = "你是用户,请给出否定回答，不要扩充回答，回答中不要提到其他需求。" + "\n\nquestion:" + param_1 + "\nanswer:"

    elif s_type == "user_profile":
        question = "根据用户" + demand + "方面的需求，结合给出的知识，反向推理用户的信息，并从可选信息的选项中选择一个最合适的选项，最后将选项用中括号括起来" + \
                   "\n用户需求：" + param_1[0] + "(" + param_2[param_1[0]].annotation + "):" + param_1[1] + \
                   "\n知识：" + extra_1.information[extra_2] + \
                   "\n可选信息及选项：" + extra_2 + ":" + str(extra_1.user_info[extra_2])
        gpt_type = "user_profile"
        think_process = f"我需要根据用户在{demand}方面的需求'{param_1[1]}'，结合领域知识'{extra_1.information[extra_2]}'，推断用户的个人信息。系统提供了可能的用户信息选项：{extra_1.user_info[extra_2]}，我需要选择最符合用户需求的选项。这种用户画像推断可以帮助系统更好地理解用户的背景和偏好，提供更个性化的服务。"
    res,ntoken = chat(question)
    return res, gpt_type, think_process,ntoken


def chat(question):
    """
    HTower API对话接口
    
    参数:
        question: 要发送给GPT的问题文本
    
    返回:
        tuple: (回答内容, 消耗的token数)
    """
    model_name='gpt-4-turbo'  # HTower推荐使用的模型
    api='your-api-key-here'  # 请替换为HTower API密钥
    url='https://api.htower.com/v1'  # HTower API端点
    client = OpenAI(base_url=url, api_key=api)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": question}
        ],
    )

    if completion.choices[0].message.content:
        print('token',completion.usage.completion_tokens)
        return completion.choices[0].message.content, completion.usage.completion_tokens
    else:
        return ""

