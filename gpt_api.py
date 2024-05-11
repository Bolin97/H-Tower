import json
import random
import default_data
import requests

from prompt import select_not_exist, select_exist, judge, token_infer, domain_infer, is_end_prompt, token_clarify, \
    sort_numbers, wrong_token, re_vague_trigger, ab_vague_trigger

EXPLICIT = default_data.EXPLICIT
RE_VAGUE_WITH_PROFILE = default_data.RE_VAGUE_WITH_PROFILE
BOUNDED = default_data.BOUNDED
RE_VAGUE_SYNONYM = default_data.RE_VAGUE_SYNONYM
AB_VAGUE = default_data.AB_VAGUE

VAGUE = default_data.VAGUE
RE_VAGUE = default_data.RE_VAGUE


def generate_sentence(param_1, param_2):
    sentence = ""
    for i in range(0, len(param_1)):
        sentence = sentence + param_1[i] + "为" + param_2[i]
        if i < len(param_1) - 1:
            sentence += ","
    return sentence


def generate_annotations(param_1, param_2):
    sentence = ""
    for i in range(0, len(param_1)):
        sentence = sentence + param_1[i] + "表示" + param_2[i]
        if i < len(param_1) - 1:
            sentence += ","
    return sentence


def generate_examples(options):
    result = ",".join(options)
    return result


def generate_history(param_2):
    result = ""
    for sen in param_2:
        result = "\n" + result + sen
    return result


def generate_demand(param_1, param_2, extra_1):
    result = ""
    for i in range(0, len(param_1)):
        result = result + str(param_1[i]) + "(" + str(extra_1[i]) + "):" + str(param_2[i]) + "\n"
    return result


def formulate_gpt_question(demand, param_1, param_2, s_type, extra_1=None, extra_2=None, extra_3=None):
    question = "你好"
    gpt_type = EXPLICIT
    # agent
    # 如果类型是问句
    if s_type == "question":
        result = generate_examples(extra_1)
        question = "你是客服，你正在与用户聊天，你们已经聊了一会了。用户想要" + demand + "，你应该怎么向他询问关于\"" + param_1 + "\"的需求信息，\"" + param_1 + "\"指\"" + param_2 + \
                   "\"，例如\"" + result + "\"请构造一个面向客户的询问语句，询问语句之后不需要提供可选项，你的回答应该只有询问语句"
        gpt_type = 'question'
    # 重新询问
    elif s_type == "retell":
        result = generate_examples(extra_1)
        question = "你是客服，你正在客户与对话，帮助他" + demand + "客户对于\"" + param_1 + "\"的要求是\"" + param_2 + \
                   "\"，你不明白用户的具体要求，希望用户能重新详细叙述他对于\"" + param_1 + "\"的要求.\"" + param_1 + "\"指\"" \
                   + extra_2 + "\"，例如\"" + result + "\"。你作为客服应该怎么说。不要有多余的语句。"
        gpt_type = "重新询问"
    # 辅助确认
    elif s_type == "assist_options":
        result = generate_examples(extra_1)
        question = "你是客服，你正在客户与对话,帮助他" + demand + "客户对于\"" + param_1 + "\"的要求是\"" + param_2 + \
                   "\"。\n(注释:" + param_1 + "是指" + extra_2 + ")\n你认为用户的具体要求不清晰，所以你想在下一次询问中给出客户几个可供选择的选项，选项为\"" + result + \
                   "\"。请构造下一次询问的问句。"
        gpt_type = "选项型辅助确认"
    elif s_type == "assist_numeric":
        result = generate_history(param_2)
        question = "你是客服，你正在客户与对话,帮助他" + demand + "，你正在确定用户对于\"" + param_1 + "\"的需求。\n" + \
                   "(注释:均价是指房屋的每月价格，购房或者租房都可以使用均价)" + \
                   "以下是你们之前的聊天记录：" + result + "你认为用户的具体要求不清晰，所以你想向他询问他的需求是高于、低于还是等于" + \
                   +extra_1 + "请接着构造问句。\n客服:"
        gpt_type = "数值型辅助确认"
    # 辅助确认+重述
    elif s_type == "assist_and_retell":
        result = generate_examples(extra_1)
        question = "你是客服，你正在客户与对话，用户想要" + demand + "客户对于\"" + param_1 + "\"的要求是\"" + param_2 + \
                   "\"。\n(注释:" + param_1 + "是指" + extra_2 + ")\n你认为用户的具体要求不清晰，所以你想在下一次询问中给出客户几个可供选择的选项，选项为\"" + result + \
                   "\"。请构造下一次询问的问句。" + \
                   "并告知用户若选项都不是他的需求请详细描述"
        gpt_type = "辅助确认+重述"
    # 澄清
    elif s_type == "clarify":
        question = "你是客服，你正在与用户对话,用户想要" + demand + "。用户回答是\"" + extra_1 + "\",你想要询问用户，关于他回答中的\"" + param_1 + \
                   "\"需求,他是否想表达他希望\"" + param_1 + "\"为\"" + param_2 + "\"。请生成询问语句。"
        gpt_type = "澄清"
    # 开放式询问
    elif s_type == "open_ended_question":
        ex = generate_examples(param_1)
        an = generate_annotations(param_1, param_2)
        question = "你是客服，你正在与用户对话,用户想要" + demand + "。你想知道用户对什么方面还有需求，例如:" + ex + "\n(注释" + an + ")," + \
                   "如果用户的需求不在列举的范畴内也请他说出来，请生成一个问句。"
        gpt_type = "token_question"
    elif s_type == "clarify_token":
        question = token_clarify + param_1 + "\n feature:" + param_2 + "\n feature annotation:" + extra_1 + "\n agent:"
        gpt_type = "clarify_token"

    # user
    # 回答
    # 要加上对话历史，这样对话才有连贯性
    elif s_type == "answer":
        random_number = random.randint(0, 2)
        ques = param_1
        keys = extra_1
        values = param_2
        types = extra_3
        user_profiles = demand
        # 数值的说范围，有profile的带profile，其他的用同义词替换
        if random_number == 0:
            re = generate_demand(keys, values, extra_2)
            question = "你是顾客，你正在与客服对话.\n客服\"" + ques + "\"，你的要求是\"" + re + "请你帮我生成回答语句,不要太直接包含需求"
            "但是需要表明意图。请你只输出回答，回答不要太正式，需求描述应该符合口语习惯，不要有多余的语句不要用双引号。"
            gpt_type = RE_VAGUE
        elif random_number == 1:  # 完全清楚
            re = generate_demand(keys, values, extra_2)
            question = "你是顾客，你正在与客服对话.\n客服\"" + ques + "\"，你的要求是\"" + re + "请你帮我生成回答语句,不要太直接包含需求"
            "但是需要表明意图。请你只输出回答，回答不要太正式，需求描述应该符合口语习惯，不要有多余的语句不要用双引号。"
            gpt_type = EXPLICIT
        else:  # 完全模糊
            re = generate_demand(keys, values, extra_2)
            question = "你是顾客，你正在与客服对话.\n客服\"" + ques + "\"，你的要求是\"" + re + "请你帮我生成回答语句,不要太直接包含需求"
            "但是需要表明意图。请你只输出回答，回答不要太正式，需求描述应该符合口语习惯，不要有多余的语句不要用双引号。"
            gpt_type = AB_VAGUE

    elif s_type == "end":
        question = "你是顾客，你正在与客服对话，你想" + demand + "。客服问你\"" + param_1 + "\"，请告诉客服你已经没有别的需求了，只要输出回答语句，" + \
                   "回答不要太正式，需求描述应该符合口语习惯，不要有多余的表达。"
        gpt_type = "需求结束"

    # 触发句
    elif s_type == "trigger":
        random_number = random.randint(0, 10)  #
        # 精准
        if 0 <= random_number < 2:
            re = generate_demand(param_1, param_2, extra_1)
            question = "你是顾客，你正在与客服对话，请生成一句话作为开场白，表示你想要" + demand + "，你的需求有:\n" + re + \
                       "注意：开场白不要太正式，需求描述应该符合中文口语习惯。"
            gpt_type = EXPLICIT
        # 同义词替换
        elif 2 <= random_number < 4:
            re = generate_demand(param_1, param_2, extra_1)
            question = "你是顾客，你正在与客服对话，请生成一句话作为开场白。首先你对需求列表中的键值进行一些联想(think step by " \
                       "step)，将需求列表中具体的值改写成描述性的话或者同义词，然后再生成开场白。开场白语气不要太正式，" + \
                       "需求描述应该符合中文口语习惯。" + re_vague_trigger + demand + "\n需求:" + re + "\nthink step by step:"

            gpt_type = RE_VAGUE_SYNONYM
        # 完全模糊
        elif 4 <= random_number < 6:
            re = generate_demand(param_1, param_2, extra_1)
            question = "你是顾客，你正在与客服对话，请生成一句话作为开场白。首先你对需求列表中的键值进行一些联想(think step by " \
                       "step)，将需求列表中具体的值改写成模糊的描述性的话。然后再生成开场白。开场白语气不要太正式，" + \
                       "需求描述应该符合中文口语习惯。" + ab_vague_trigger + demand + "\n需求:" + re + "\nthink step by step:"
            gpt_type = AB_VAGUE
        # 不带需求
        else:
            question = "你是顾客，你正在与客服对话，请生成一句话作为开场白，表示你想要" + demand + "开场白不要太正式应该符合口语习惯"
            gpt_type = "不含需求"
    elif s_type == "no_interest":
        question = "客服:" + param_1 + "\n用户:我对这方面没有需求。" + "\n改写用户的话，使其对客服的问题有针对性。注意不要提到别的需求" + \
                   "用户:"
        gpt_type = "no interest"
    # 需求判断句
    elif s_type == "judge":
        question = judge + param_1 + "\"\nagent infer:" + extra_1 + "\nuser demand:" + param_2 + "\nuser:"
        gpt_type = "需求判断"
    # 需求辅助确认
    elif s_type == "select":
        if extra_1 == "selection":
            question = select_exist + "客服:\"" + param_1 + "\"\n用户需求:" + param_2 + "\n用户:"
            gpt_type = "存在于选项中"
        elif extra_1 == "not_exist":
            question = select_not_exist + "客服:\"" + param_1 + "\"\n用户:"
            gpt_type = "不存在于选项中"

    # 功能性
    elif s_type is None:
        question = "\"" + param_1 + "\"。这个句子表示赞同还是反对。"
    # 推测token
    elif s_type == "token_infer":
        if param_2 is None:
            question = "<feature>" + extra_1 + token_infer + param_1 + "\ncertain_label:None" + "\nlabel:"
        else:
            tokens = generate_examples(param_2)
            question = "<feature>" + extra_1 + token_infer + param_1 + "\ncertain_label:" + tokens + "\nlabel:"
    # 判断对话是否结束
    elif s_type == "is_end":
        question = is_end_prompt + param_1 + "\n answer:" + param_2 + "\noptions:\nA:用户没有额外的需求了\nB:用户还有需求" + "\nresult:"

    # 判断用户需求属于哪个领域和子类
    elif s_type == "domain_infer":
        question = domain_infer + param_1 + "\ndomains:" + param_2 + "\nresult:"
        gpt_type = "domain_infer"
    elif s_type == "domain_confirm":
        sentence = "、".join([f"“{option}”" for option in param_1[:-1]]) + f"或者“{param_1[-1]}”"
        question = "用户:\"" + param_2 + "\"\n 询问用户需求涉及的是" + sentence + "。不要询问用户的具体需求。"
        gpt_type = "domain_confirm"
    elif s_type == "domain_answer":
        options = generate_examples(extra_1)
        question = "客服的问题\"" + param_1 + "\"\n 客服问题涉及的选项:" + options + "\n 你的答案:" + param_2 + "\n 请根据客服的问题生成回答，" + \
                   "回答不要太正式，要符合口语习惯。"

    elif s_type == "numeric_result":
        question = "客服:" + param_1 + "\n用户:" + param_2 + "\n以上对话中，用户的需求是什么：\nA：低于客服给出的数值\n" + \
                   "B：等于客服给出的数值\n+C：高于客服给出的数值\n请给出选项"
    # 生成同意或者不同意的句子
    elif s_type == "agree":
        question = "针对question给出肯定回答，回答要与question连贯，不要扩充回答。" + "\n\nquestion:" + param_1 + "\nanswer:"
    elif s_type == "disagree":
        question = "针对question给出否定回答，回答要与question连贯，不要扩充回答。" + "\n\nquestion:" + param_1 + "\nanswer:"
    elif s_type == "sort":
        numbers = generate_examples(param_1)
        question = sort_numbers + numbers + "\n列表结果:"
        gpt_type = "sort"
    elif s_type == "user_profile":
        question = "根据用户" + demand + "方面的需求，结合给出的知识，反向推理用户的信息，并从可选信息的选项中选择一个最合适的选项，最后将选项用中括号括起来" + \
                   "\n用户需求：" + param_1[0] + "(" + param_2[param_1[0]].annotation + "):" + param_1[1] + \
                   "\n知识：" + extra_1.information[extra_2] + \
                   "\n可选信息及选项：" + extra_2 + ":" + str(extra_1.user_info[extra_2]) + \
                   "\n逐步推理及选择："
        gpt_type = "user_profile"
    # verify
    elif s_type == "not_required_token":
        question = "你想要" + demand + "你正在和客服聊天。\n" + "客服：以下是通过刚刚的对话我所了解的需求，请您核对以下的列表，如果有错误或者遗漏请及时告诉我。\n" + \
                   + param_1 + ":" + param_2 + "\n客服错误的认为你对\"" + param_1 + "\"有需求,请告诉他你对\"" + param_1 + "\"没有需求。"
    elif s_type == "ask_for_verify":
        question = "改写下面这段话，表示赞同，并询问他是否还认为有可以改进的地方:\n" + param_1
    elif s_type == "missing_token_value":
        question = "你想要" + demand + "你正在和客服聊天。\n" + "客服：以下是通过刚刚的对话我所了解的需求，请您核对以下的列表，如果有错误或者遗漏请及时告诉我。\n[列表内容]" + \
                   + "你之前说过你对\"" + param_1 + "\"有需求，但是客服遗漏了你对\"" + param_1 + "\"的需求，请生成一句话告诉客服你对\"" + param_1 + "\"有需求" + \
                   "(不要说明具体\"" + param_1 + "\")"
    elif s_type == "wrong_token":
        question = "用户正在和客服聊天。请你根据错误内容生成回答，告诉客服他的列表中的信息有误。" + \
                   "\n客服：以下是通过刚刚的对话我所了解的需求，请您核对以下的列表，如果有错误或者遗漏请及时告诉我。" + \
                   "\n[列表内容]" + wrong_token + demand + "\n错误内容:" + param_1 + ":" + param_2 + "\n用户:"

    url = "https://oa.api2d.net/v1/chat/completions"
    messages = [
        {
            'role': 'user',
            'content': question,
        }]
    payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "safe_mode": False
    })
    headers = {
        'Authorization': 'Bearer fk208078-xxxxxxxxxxxxxxxx',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    res = response.json()
    res = res["choices"][0]["message"]["content"]
    return res, gpt_type
