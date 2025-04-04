import json
import random

EXPLICIT = "精准"
RE_VAGUE_WITH_PROFILE = "相对模糊(参考user profile)"
RE_VAGUE_SYNONYM = "相对模糊(同义词)"
RE_VAGUE = "相对模糊"
AB_VAGUE = "完全模糊"
BOUNDED = "范围回答"
VAGUE = "模糊询问"


# 需求列表的格式
class DEMAND_LIST:
    def __int__(self, trigger_features=None, domain=None, demand_type=None):
        self.id = 0  # 自增字段,需求列表的id
        self.user_profile = {}  # 与当前需求相关的用户画像
        self.trigger = None  # 触发句
        self.trigger_features = trigger_features  # 触发句所描述的feature
        self.trigger_type = None  # 触发句类型
        self.domain = domain  # 用户需求所属领域
        self.demand_type = demand_type  # 用户需求在该领域的类别
        self.features = []  # 用户需求包含的所有feature
        self.page_number = None  # 至少来源于某一个网页
        self.need = {}  # 需求列表


# 网页
class WEBPAGE:
    def __init__(self, index=None, image=None, website=None, web_class=None, class_type=None, required_tokens=None,
                 tree=None, keys=None):
        self.index = index  # 网页在文件中的index，从1开始
        self.image = image  # 网页对应的dashboard截图
        self.website = website  # 网页对应的网站地址
        self.web_class = web_class  # 网页的分类（大分类）
        self.class_type = class_type  # 网页的子类
        self.required_tokens = required_tokens
        self.tree = tree or []  # 网页的kv对
        self.keys = keys or []  # 网页的key集合


# token information
class QTOKEN:  # token (拥有某同义key的一类页面)
    def __init__(self):
        # self.token = None  # 同义key
        self.token_type = None  # token的类型，数值型或者选项型
        self.TF = 0,  # 在所有的网页中出现的次数
        self.annotation = None  # 解释
        self.average_item_num = 0  # 每个网页该key下value数的平均
        #  self.history = []  # 关于该token的聊天记录
        self.pages = []  # 网页索引及在对应网页key的称呼


# conversation
class CONVERSATION:
    def __init__(self):
        self.role = None  # 角色
        self.sentence = None  # 生成的句子
        self.think_process= None
        self.type = None  # 句子类型
        self.target_feature = None  # agent询问的feature
        self.feature_value = {}  # 句子涉及的token
        self.tolerance =None #用户忍耐度
        self.prospect=None #期望函数取值


class DEMAND:  # 需求
    def __init__(self):
        self.page_number = None  # 满足用户需求的网页索引
        self.key = []  # key值
        self.token = []  # key 对应的token
        self.value = []  # key对应的value


class PAGEKEY:
    def __init__(self):
        self.token = None
        self.page_num = None  # 对应第几个网页
        self.key = []  # 一个网页中可能有同义的token
        self.amount = 0  # value的个数


# 用户画像(用户)
class USERPROFILE:
    def __init__(self, token=None, user_info=None, information=None):
        self.token = token
        self.user_info = user_info
        self.information = information


# 用户画像，agent所知道的
class USE_USERPROFILE:
    def __init__(self, key=None, value=None):
        self.profile_key = key
        self.profile_value = value
        self.profile_info = None


class PROCESSINGTOKEN:
    def __init__(self):
        self.infer_value = None  # 当前的infer_value
        self.user_value = None  # 正确的值
        self.value_type = None  # value的类型(选项或数值或范围)
        self.dialog_history = []  # 关于这个value的对话历史
        self.options = None  # 这个value的可选项，选项型和数值型是列表，范围型是字典
        self.unsuccessful = 0  # 推理失败的次数
        self.interested = None  # 用户此次对话是否提到该feature
        self.token_answer_type = None  # 用户回答的类型


def read_token(token_averages, domain_path, TF=None, valid_tokens=None, valid_page_numbers=None):
    path = "data/" + domain_path + "/" + domain_path + "_token_tf.json"
    with open(path, "r", encoding="utf-8") as file:
        tf = json.load(file)

    # 将TOKEN数据存储为QTOKEN类的实例列表
    qtoken_list = {}
    for item in tf:
        if valid_tokens is None or item["token"] in valid_tokens:
            qtoken = QTOKEN()
            qtoken.token_type = item["token_type"]
            if TF is not None:
                qtoken.TF = TF[item["token"]]
            else:
                qtoken.TF = item["TF"]
            qtoken.annotation = item["annotation"]
            qtoken.average_item_num = token_averages[item["token"]]
            if valid_page_numbers is not None:
                for page_t in item["pages"]:
                    if page_t["page_number"] in valid_page_numbers:
                        qtoken.pages.append(page_t)
            else:
                qtoken.pages = item["pages"]
            qtoken_list[item["token"]] = qtoken
    return qtoken_list


def read_domain():
    # path to web中存储着网页的类型信息和在相关领域data文件夹下的路径信息
    with open("data/path_to_web.json", "r", encoding='utf-8') as file:
        data = json.load(file)
    domains = []
    types = []
    path = []
    for domain in data:
        domains.append(domain["domain"])
        types.append(domain["type"])
        path.append(domain["path"])
    return domains, types, path
