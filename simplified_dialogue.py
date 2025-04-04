import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from agent_gpt import formulate_agent_question, formulate_user_response
from agent2agent import extract_feature_values, UserType
from openai import OpenAI
#agent对话特征提取大模型api
api_extract_agent = ''
url_extract_agent = ''
model_extract_agent=''

#user对话特征提取大模型api
api_extract_user = ''
url_extract_user = ''
model_extract_user=''

#agent对话大模型api
api_agent = ''
url_agent = ''
model_agent=''

#user对话大模型api
api_user = ''
url_user = ''
model_user=''

con_nums=10 #最大限制对话轮数
user_type='normal' #用户类型选择:'normal','exponential','expectation'
@dataclass
class CONVERSATION:
    role: str = ""
    sentence: str = ""
    s_type: str = ""
    target_feature: list = ()
    feature_value: Dict[str, str] = field(default_factory=dict)
    think_process: Optional[str] = None
    tolerance: Optional[float] = None
    prospect: Optional[float] = None
    termination_value: Optional[float] = None


@dataclass
class DEMAND:
    key: List[str] = field(default_factory=list)
    token: List[str] = field(default_factory=list)
    value: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            'key': self.key,
            'token_value': {token: value for token, value in zip(self.token, self.value)}
        }

    def __json__(self):
        return self.to_dict()


class SimplifiedDialogueSystem:
    def __init__(self, user_type: str = user_type, max_time_minutes: int = 15): #用户类型选择
        self.current_conversations = []
        self.user_type = UserType(user_type)  # 创建用户类型实例
        self.interactions_rounds = 0
        self.user_demand = DEMAND()
        self.remain_demand = {}
        self.matched_count = 0
        self.completeness_rate = 0
        self.redundancy_rate = 0
        self.agent_demands = DEMAND()
        self.start_time = time.time()
        self.max_time_minutes = max_time_minutes

    def simple_token_count(self, text):
        # 简单分词：中文按字，英文按单词
        import re
        # 中文按字分割
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        # 英文按单词分割
        english_words = re.findall(r'[a-zA-Z]+', text)
        # 标点符号等
        punctuations = re.findall(r'[^\w\s]', text)
        return len(chinese_chars) + len(english_words) + len(punctuations)

    def should_terminate(self) -> bool:
        if not self.user_type:
            return False

        # 调用UserType的should_end_conversation方法获取终止判断
        should_end, value = self.user_type.should_end_conversation()
        return should_end

    def extract_agent_features(self, agent_question: str, need_keys: List[str]) -> List[str]:
        """
        从agent的问题中提取询问的特征。

        Args:
            agent_question (str): agent的问题
            need_keys (List[str]): 用户需求的特征列表

        Returns:
            List[str]: 提取出的特征列表，如果特征在need_keys中存在则使用need_keys中的名称
        """
        # 调用API
        client = OpenAI(base_url=url_extract_agent, api_key=api_extract_agent)

        # 构建提示词
        prompt = f"""你是一个特征提取专家，请分析以下客服问题中询问了哪些特征：

        客服问题: {agent_question}

        用户需求特征列表: {need_keys}

        请提取客服问题中询问的所有特征，如果特征在用户需求特征列表中有对应项，则使用列表中的名称；如果没有对应项，则使用提取出的特征名称。
        返回特征为元素,例如:'写字楼面积',返回'面积',
        例如：
        - 如果客服问题是"您好，您对车辆的驱动方式和车龄有什么要求吗？"，而用户需求特征列表中有"驱动"但没有"车龄"，则应返回["驱动","车龄"]

        请以JSON格式返回提取的特征列表，格式为 ["特征1","特征2", ...]
        只返回JSON数组，不要返回其他文字。
        """

        response = client.chat.completions.create(
            model=model_extract_agent,
            messages=[
                {"role": "system", "content": prompt}
            ]
        )
        # 获取结果
        result = response.choices[0].message.content
        try:
            # 尝试解析返回的JSON
            extracted_features = json.loads(result)
            return extracted_features
        except json.JSONDecodeError:
            # 如果解析失败，尝试从文本中提取JSON部分
            try:
                # 查找可能的JSON部分
                json_start = result.find('[')
                json_end = result.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = result[json_start:json_end]
                    extracted_features = json.loads(json_str)
                    return extracted_features
            except:
                pass

            # 如果仍然失败，返回空列表
            print("无法从大模型回答中提取JSON格式的特征列表")
            return []

    def extract_user_features(self, agent_question: str, user_answer: str, need_keys) -> Dict[str, str]:
        """
        从用户回答中提取特征及其值。

        Args:
            agent_question (str): agent的问题 (可以为空字符串 "" 如果是trigger 阶段)
            user_answer (str): 用户的回答
            need_keys (List[str]): 用户需求的特征列表

        Returns:
            Dict[str, str]: 提取出的特征及其值的字典
        """
        # 调用API

        prompt = f"""你是一个特征提取专家，请分析以下对话，提取用户回答中关于特定特征的值：

        客服问题: {agent_question}
        用户回答: {user_answer}

        需要提取的特征列表: {need_keys}

        请分析用户回答中是否包含了特征列表中的特征信息，如果包含，请提取出特征名称及其对应的值。
        范围类别的value，其中的"到"替换为"-"，示例："30到50万字"应该输出为"30-50万字"。
        提取的值为元素,示例:'在4号线附近最好',得到的值应该为'4号线';
        请以JSON格式返回每个特征对应的值，格式为 {{"特征1": "值1", "特征2": "值2", ...}}。
        只用返回有值的特征,没有值的不要返回,'无特殊要求'为没有值
        只返回JSON对象，不要返回其他文字。
        """
        if not agent_question:  # modify prompt for trigger sentence
            prompt = f"""你是一个特征提取专家，请分析以下用户初始语句，提取用户语句中关于特定特征的值：

        用户初始语句: {user_answer}

        需要提取的特征列表: {need_keys}

        请分析用户语句中是否包含了特征列表中的特征信息，如果包含，请提取出特征名称及其对应的值。
        范围类别的value，其中的"到"替换为"-"，示例："30到50万字"应该输出为"30-50万字"。
        只返回有值的特征,
        请以JSON格式返回每个特征对应的值，格式为 {{"特征1": "值1", "特征2": "值2", ...}}。
        只返回JSON对象，不要返回其他文字。
        """

        client = OpenAI(base_url=url_extract_user, api_key=api_extract_user)
        response = client.chat.completions.create(
            model=model_extract_user,
            messages=[
                {"role": "system", "content": prompt}
            ]
        )
        # 获取结果
        result = response.choices[0].message.content
        # response, response_type, think_process, annotation = formulate_user_response(prompt)
        response_type = 'anwser'
        '''response = client.chat.completions.create(
            model="doubao-1-5-lite-32k-250115",
            messages=[
                {"role": "system", "content": "你是一个特征提取专家，擅长从对话中提取关键信息。"},
                {"role": "user", "content": prompt}
            ]
        )

        # 获取结果
        result = response.choices[0].message.content'''

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

    def save_conversation(self, role: str, sentence: str, s_type: str,
                          target_feature: Optional[str] = None,
                          user_tokens: Optional[List[str]] = None,
                          user_values: Optional[List[str]] = None,
                          think_process: Optional[str] = None,
                          tolerance: Optional[float] = None,
                          prospect: Optional[float] = None,
                          extracted_features={}) -> None:
        """
        记录对话信息。
        Args:
            role: 对话角色("user" 或 "agent")
            sentence: 对话内容
            s_type: 对话类型("trigger", "question", "answer" 等)
            target_feature: 目标特征
            user_tokens: 用户回答中的特征标记
            user_values: 用户回答中的特征值
            think_process: 思考过程
            tolerance: 容忍度
            prospect: 前景预期
            annotation: 标注信息。Defaults to None.
        """
        conversation = CONVERSATION()
        conversation.role = role
        conversation.sentence = sentence
        conversation.s_type = s_type

        conversation.think_process = think_process
        conversation.tolerance = tolerance
        conversation.prospect = prospect

        # 当角色是用户时，获取终止值和函数值
        if role == "user":
            should_end, value = self.user_type.should_end_conversation()
            conversation.termination_value = value
            # 保存函数值
            if self.user_type.type_name in ["exponential", "expectation"]:
                conversation.function_value = value

        self.interactions_rounds += 1

        # 当角色是agent时，使用extract_agent_features提取问题中的特征
        if role == "agent":
            # 提取agent问题中的特征
            extracted_values = self.extract_agent_features(sentence, list(self.remain_demand.keys()))
            self.matched_count -= len(extracted_values)
            self.user_type.update_key_counts(self.matched_count)
            # 将agent的问题保存到对话记录中
            for i in extracted_values:
                self.agent_demands.key.append(i)
            conversation.target_feature = extracted_values

        # 当角色是用户时，使用extract_user_features提取回答中的特征值
        if role == 'user':
            if user_tokens:
                # 更新特征值字典
                for i, token in enumerate(user_tokens):
                    conversation.feature_value[token] = user_values[i]

                # 如果有明确的user_tokens和user_values，也添加到feature_value中
                if user_tokens is not None:
                    for i, token in enumerate(user_tokens):
                        if user_values is not None and i < len(user_values):
                            conversation.feature_value[token] = user_values[i]
                        else:
                            conversation.feature_value[token] = ""

            # 将用户的回答保存到对话记录中
        self.current_conversations.append(conversation)

    def get_conversation_history(self) -> str:
        """
        获取对话历史记录。
        """
        history = []
        for conv in self.current_conversations:
            history.append(f"{conv.role}: {conv.sentence}")
        return "\n".join(history)

    def agent_turn(self, c) -> str:
        """
        Agent回合，生成问题。
        """
        if c > 5:
            a = "你根据历史记录,如果觉得已经确定了用户所有需求,可以回答'结束对话'来结束对话"
        else:
            a = ''
        conversation_history = self.get_conversation_history()
        prompt = f"""你是一个负责询问用户需求的助手。根据对话历史，提出问题以确认用户的需求。
                        可能的需求特征包括：A.书籍类用户可能需求列表:1.分类(作品的分类), 2.状态(作品的连载状态), 3.属性(作品的属性), 4.字数(作品的字数), 5.品质(作品的品质), 6.更新时间(作品的更新时间范围), 7.标签(作品的标签)
        B.车辆类用户可能需求列表:1.品牌(车辆的品牌), 2.价格(车辆的价格), 3.级别(车辆的级别), 4.能源(车辆的能源类型), 5.里程(车辆已经跑过的里程), 6.车龄(车辆被使用过的年限), 7.变速箱(变速箱类型), 8.排量标准(车辆的排量标准) 9.排量(车辆的排量), 10.驱动(车辆的驱动方式), 11.厂商属性(汽车的厂商属性), 12.懂车分(懂车分是汽车评价系统5分代表优秀。4分到5分表示表现良好。3分到4分表示中规中矩1.5分到2.5分表示缺点明显), 13.续航(电池续航里程), 14.国别(车辆的生产国家), 15.结构(车身结构), 16.座位数(车辆的座位数), 17.气缸数(车辆的气缸数), 18.进气方式(车辆的进气方式), 19.安全配置(车辆的安全配置), 20.舒适配置(车辆的舒适配置)
        C.房屋类用户可能需求列表:1.区域(房屋所在的区域位置), 2.地铁(房屋附近的地铁线路), 3.总价(房屋的总价范围，一般在购房时使用总价), 4.单价(房屋的每月价格，购房或者租房都可以使用单价), 5.面积(房屋的面积范围), 6.户型(房屋的房间数量或居室规模), 7.特色(房屋附加选项或特色), 8.朝向(房屋的朝向), 9.楼层(房屋的处在的楼层), 10.房龄(房屋建造的年龄范围), 11.物业品牌(房源的品牌物业), 12.产权(房屋的所有权类型), 13.装修(房屋装饰和装潢的各种程度和类型), 14.建筑类别(房屋的建筑类型), 15.方式(租房的不同选择或形式), 16.房屋类型(房产的种类或类型), 17.距离(房屋离地铁站的距离)
        D.写字楼类用户可能需求列表:1.区域(写字楼所在的区域位置), 2.租金(写字楼的价格范围), 3.面积(写字楼的面积范围), 4.房屋类型(写字楼类型), 5.特色(写字楼的特色), 6.更多找房条件(更多找房条件), 7.地铁(写字楼附近的地铁线路), 8.距离(写字楼距离地铁站的距离), 9.来源(选择写字楼的信息来源), 10.销售状态(写字楼的销售状态)
        E.影视类用户可能需求列表:1.地区(制片的区域), 2.类型(影视作品的类型), 3.分类(影视作品的类别), 4.规格或版权(影片的规格或版权), 5.年份(影片的年份), 6.付费类型(影片的付费类型), 7.排序方式(影片的排序方式), 8.影片类型推荐(推荐的影片类型), 9.演员(影片的演员), 10.节目(影片的节目), 11.连载情况(影片的连载情况), 12.年龄分类(影片的年龄分类)。
        ** 目标是确认用户的需求,你只能询问上面有的特征,问题简洁直白**
        **你一次最多只能询问两个特征,所以需要分析得到最佳的特征提问顺序**
        **不能一次提问过多特征**
        以下为你与用户对话历史记录{conversation_history},{a}"""

        '''from zhipuai import ZhipuAI
        client = ZhipuAI(api_key='')  
        response = client.chat.completions.create(
            model="glm-4-flash",  
            messages=[
                {"role": "user", "content": prompt},
            ],
        )'''
        client = OpenAI(base_url=url_agent, api_key=api_agent)
        response = client.chat.completions.create(
            model=model_agent,
            messages=[
                {"role": "system", "content": prompt}
            ]
        )
        # 获取结果

        question = response.choices[0].message.content
        # question, think_process, _ = formulate_agent_question(prompt)

        # 更新token计数
        token_count = self.simple_token_count(question)
        self.user_type.update_tokens(token_count)
        think_process = ''

        return question, think_process

    def get_dialogue_stats(self, demand_index: int) -> Dict:
        # 计算完成率
        total_demands = len(self.user_demand.token)
        matched_values = sum(1 for token, value in zip(self.user_demand.token, self.user_demand.value)
                             if token in self.agent_demands.token and value in self.agent_demands.value)
        self.completeness_rate = matched_values / total_demands if total_demands > 0 else 0.0
        matched_counts = 0
        # 计算冗余率
        for z in self.agent_demands.key:
            if z in self.user_demand.token:
                matched_counts += 1
        total_interaction = len(self.agent_demands.key)
        self.redundancy_rate = (
                                       total_interaction - matched_counts) / total_interaction if total_interaction > 0 else 0.0

        return {
            "demand_index": demand_index,
            "conversations": [
                {
                    "role": c.role,
                    "sentence": c.sentence,
                    "type": c.s_type,
                    "target_feature": c.target_feature,
                    "token_value": c.feature_value,
                    "termination_value": c.termination_value if hasattr(c, 'termination_value') else None,
                    "function_value": c.function_value if hasattr(c, 'function_value') else None
                } for c in self.current_conversations
            ],
            "interaction_rounds": self.interactions_rounds // 2,
            "matched_demand_count": self.matched_count,
            "completeness_rate": round(self.completeness_rate, 2),
            "redundancy_rate": round(self.redundancy_rate if self.interactions_rounds > 0 else 0.0, 2),
            # avoid negative redundancy rate at start
            "agent_demands": self.agent_demands.to_dict(),  # 添加 .to_dict()
            "original_user_need": {
                "keys": self.user_demand.token,
                "token-values": {token: value for token, value in zip(self.user_demand.token, self.user_demand.value)},
            }
        }

    def user_turn(self, agent_question: str, remain_needs: List[str], trigger_sentence) -> str:
        """
        User回合，根据剩余需求和agent问题生成回答。

        Args:
            agent_question: Agent的问题
            remain_needs: 剩余未匹配的需求列表

        Returns:
            用户的回答
        """
        # 构造用户回答的prompt
        prompt = f"""**你的身份是一名客户，有特定的需求但不会主动提出。**
        **重要:只回答被询问到的需求,不要在回答中添加没有问到的剩余需求**
        如果助手的问题与你的需求无关，可以简单回答你没有特定要求。
        你的目的:{trigger_sentence}
        你的剩余需求是：{remain_needs}。
        客服问题:{agent_question},
        **不要回答没有被询问到的剩余特征需求,不要直接回答没有被问到的需求**
        请根据助手的问题，自然地回答与你需求相关的内容
        """

        client = OpenAI(base_url=url_user, api_key=api_user)
        response = client.chat.completions.create(
            model=model_user,
            messages=[
                {"role": "system", "content": prompt}
            ]
        )
        # 获取结果
        response = response.choices[0].message.content
        response_type = 'anwser'
        # 提取用户回答中的特征值
        if len(remain_needs) > 0:
            # 提取用户回答中的特征和值
            extracted_values = self.extract_user_features(agent_question, response, list(remain_needs.keys()))

            if extracted_values:
                # 更新匹配计数
                self.matched_count += len(extracted_values)

                # 从remain_needs中移除已匹配的特征
                for feature in extracted_values.keys():
                    if feature in self.remain_demand.keys():
                        self.remain_demand.pop(feature)
                        # 更新用户类型的特征计数

                # 更新用户类型的需求状态
                self.user_type.update_key_counts(self.matched_count)
                # 更新agent_demands对象
                for feature, value in extracted_values.items():
                    # 检查feature是否已存在于agent_demands.key中
                    if feature not in self.agent_demands.key:
                        self.agent_demands.key.append(feature)
                        self.agent_demands.token.append(feature)
                        self.agent_demands.value.append(value)
                    else:
                        # 如果已存在，更新对应的value
                        if feature not in self.agent_demands.token:
                            self.agent_demands.token.append(feature)
                            self.agent_demands.value.append(value)
                        else:
                            index = self.agent_demands.token.index(feature)
                            self.agent_demands.value[index] = value

                    # 从remain_needs中移除已匹配的特征
                    if feature in remain_needs:
                        remain_needs.remove(feature)

                # 保存对话状态，包含提取到的特征值
                self.save_conversation("user", response, response_type,
                                       user_tokens=list(extracted_values.keys()),
                                       user_values=list(extracted_values.values()))
            else:
                # 即使没有提取到特征值，也要保存对话记录
                self.save_conversation("user", response, response_type)
        else:
            # 没有剩余需求时，仍然保存对话记录
            self.save_conversation("user", response, response_type )

        return response


def load_test_data(file_path: str) -> List[Dict]:
    """从JSON文件加载测试数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_dialogue_test(test_case: Dict, case_index: int, total_cases: int) -> Dict:
    """运行单个对话测试"""
    dialogue_system = SimplifiedDialogueSystem()

    # 设置用户需求
    features = test_case.get('features', [])
    need = test_case.get('need', {})

    dialogue_system.user_demand.key = features
    dialogue_system.user_demand.token = list(need.keys())
    dialogue_system.user_demand.value = list(need.values())

    dialogue_system.remain_demand = need
    # 保存用户触发语句到对话历史
    trigger_sentence = test_case.get('trigger', '')
    if trigger_sentence:
        extracted_values = dialogue_system.extract_user_features('', need_keys=need.keys(),
                                                                 user_answer=trigger_sentence)
        if extracted_values:
            # 更新匹配计数
            dialogue_system.matched_count += len(extracted_values)

            # 从remain_needs中移除已匹配的特征
            for feature in extracted_values.keys():
                if feature in dialogue_system.remain_demand.keys():
                    dialogue_system.remain_demand.pop(feature)

            # 更新用户类型的需求状态
            dialogue_system.user_type.update_needs(len(dialogue_system.user_demand.token),
                                                   dialogue_system.matched_count)

            # 更新agent_demands对象
            for feature, value in extracted_values.items():
                # 检查feature是否已存在于agent_demands.key中
                if feature not in dialogue_system.agent_demands.key:
                    dialogue_system.agent_demands.key.append(feature)
                    dialogue_system.agent_demands.token.append(feature)
                    dialogue_system.agent_demands.value.append(value)
                else:
                    # 如果已存在，更新对应的value
                    if feature not in dialogue_system.agent_demands.token:
                        dialogue_system.agent_demands.token.append(feature)
                        dialogue_system.agent_demands.value.append(value)
                    else:
                        index = dialogue_system.agent_demands.token.index(feature)
                        dialogue_system.agent_demands.value[index] = value

                # 从remain_needs中移除已匹配的特征
                if feature in dialogue_system.remain_demand:
                    dialogue_system.remain_demand.pop(feature)
        dialogue_system.save_conversation("user", trigger_sentence, 'trigger',
                                          user_tokens=list(extracted_values.keys()),
                                          user_values=list(extracted_values.values()))

    # 设置用户特征
    dialogue_system.user_type.set_user_features(need)

    print(f"\n测试用例 [{case_index + 1}/{total_cases}]")
    print(f"初始需求: {need}")
    z = 0
    # 模拟对话流程
    while len(dialogue_system.remain_demand.keys()) > 0:
        # 检查是否应该提前结束对话
        should_end, end_value = dialogue_system.user_type.should_end_conversation()
        if should_end:
            print(f"\n用户类型 {dialogue_system.user_type.type_name} 触发提前结束对话，结束值: {end_value}")
            break
        if z > con_nums: #最大限制对话轮数
            break
        z += 1
        # Agent回合
        agent_question, think_process = dialogue_system.agent_turn(z)
        print(f"\nAgent: {agent_question}")
        dialogue_system.save_conversation("agent", agent_question, "question", think_process=think_process)
        # 检查agent是否主动结束对话
        if "结束对话" in agent_question or "对话结束" in agent_question:
            if z > 5:
                print("\nAgent主动结束对话")
                break

        # User回合
        user_response = dialogue_system.user_turn(agent_question, dialogue_system.remain_demand, trigger_sentence)
        print(f"User: {user_response}")

        # 更新用户类型统计
        dialogue_system.user_type.update_needs(len(features), dialogue_system.matched_count)

    stats = dialogue_system.get_dialogue_stats(test_case['id'])
    print(f"\n当前进度:")
    print(f"对话轮次: {stats['interaction_rounds']}")
    print(f"已匹配需求数: {stats['matched_demand_count']}")
    print(f"完成率: {stats['completeness_rate']:.2%}")
    print(f"冗余率: {stats['redundancy_rate']:.2%}")

    # 保存当前对话结果
    write_to_file([stats])

    print("\n对话完成!")
    print(f"最终匹配结果: {dialogue_system.agent_demands}")
    return dialogue_system.get_dialogue_stats(test_case['id'])


def write_to_file(result):
    """将单个需求的对话信息写入JSON文件。

    Args:
        result (dict): 包含单个需求对话信息的字典。
    """
    # 读取现有的对话数据
    try:
        with open("./data/conversations.json", "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = []

    # 添加新的对话结果
    all_results.extend(result)

    # 写入文件
    with open("./data/conversations.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)


def main():
    # 加载测试数据
    test_cases = load_test_data('data/demands.json') #需求文件地址
    total_cases = len(test_cases)

    # 存储所有测试结果
    all_results = []
    # 运行所有测试用例
    for i, test_case in enumerate(test_cases):
        dialogue_system = SimplifiedDialogueSystem()
        start = time.time()
        print(f"\n{'=' * 50}")
        print(f"开始测试用例 {i + 1}/{total_cases}")
        print(f"用户触发语: {test_case['trigger']}")

        # 运行对话测试
        result = run_dialogue_test(test_case, i, total_cases)
        all_results.append(result)
        # 计算并显示总体进度
        endd = time.time()
        print(f'剩余时间:{(endd - start) * (len(test_cases) - i) // 60}')
        progress = (i + 1) / total_cases * 100
        print(f"\n总体进度: {progress:.1f}%")

    # 输出测试统计
    total_rounds = sum(r['interaction_rounds'] for r in all_results)
    avg_rounds = total_rounds / total_cases

    print(f"\n{'=' * 50}")
    print("测试完成! 总体统计:")
    print(f"总测试用例数: {total_cases}")
    print(f"平均对话轮次: {avg_rounds:.2f}")

    # 保存对话数据到文件


# write_to_file(all_results)

if __name__ == "__main__":
    main()