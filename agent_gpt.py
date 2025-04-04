import json
from openai import OpenAI

def formulate_agent_question(conversation_history: str) -> tuple:
    """
    生成agent的问题。
    
    Args:
        conversation_history (str): 对话历史记录
        
    Returns:
        tuple: (生成的问题, 思考过程, 回答类型)
    """
    # 构建基础prompt
    prompt = f"""你是一个负责询问用户需求的助手。根据对话历史，提出问题以确认用户的需求。
    可能的需求特征包括：A.书籍类用户可能需求列表:1.分类(作品的分类), 2.状态(作品的连载状态), 3.属性(作品的属性), 4.字数(作品的字数), 5.品质(作品的品质), 6.更新时间(作品的更新时间范围), 7.标签(作品的标签)
B.车辆类用户可能需求列表:1.品牌(车辆的品牌), 2.价格(车辆的价格), 3.级别(车辆的级别), 4.能源(车辆的能源类型), 5.里程(车辆已经跑过的里程), 6.车龄(车辆被使用过的年限), 7.变速箱(变速箱类型), 8.排量标准(车辆的排量标准) 9.排量(车辆的排量), 10.驱动(车辆的驱动方式), 11.厂商属性(汽车的厂商属性), 12.懂车分(懂车分是汽车评价系统5分代表优秀。4分到5分表示表现良好。3分到4分表示中规中矩1.5分到2.5分表示缺点明显), 13.续航(电池续航里程), 14.国别(车辆的生产国家), 15.结构(车身结构), 16.座位数(车辆的座位数), 17.气缸数(车辆的气缸数), 18.进气方式(车辆的进气方式), 19.安全配置(车辆的安全配置), 20.舒适配置(车辆的舒适配置)
C.房屋类用户可能需求列表:1.区域(房屋所在的区域位置), 2.地铁(房屋附近的地铁线路), 3.总价(房屋的总价范围，一般在购房时使用总价), 4.单价(房屋的每月价格，购房或者租房都可以使用单价), 5.面积(房屋的面积范围), 6.户型(房屋的房间数量或居室规模), 7.特色(房屋附加选项或特色), 8.朝向(房屋的朝向), 9.楼层(房屋的处在的楼层), 10.房龄(房屋建造的年龄范围), 11.物业品牌(房源的品牌物业), 12.产权(房屋的所有权类型), 13.装修(房屋装饰和装潢的各种程度和类型), 14.建筑类别(房屋的建筑类型), 15.方式(租房的不同选择或形式), 16.房屋类型(房产的种类或类型), 17.距离(房屋离地铁站的距离)
D.写字楼类用户可能需求列表:1.区域(写字楼所在的区域位置), 2.租金(写字楼的价格范围), 3.面积(写字楼的面积范围), 4.房屋类型(写字楼类型), 5.特色(写字楼的特色), 6.更多找房条件(更多找房条件), 7.地铁(写字楼附近的地铁线路), 8.距离(写字楼距离地铁站的距离), 9.来源(选择写字楼的信息来源), 10.销售状态(写字楼的销售状态)
E.影视类用户可能需求列表:1.地区(制片的区域), 2.类型(影视作品的类型), 3.分类(影视作品的类别), 4.规格或版权(影片的规格或版权), 5.年份(影片的年份), 6.付费类型(影片的付费类型), 7.排序方式(影片的排序方式), 8.影片类型推荐(推荐的影片类型), 9.演员(影片的演员), 10.节目(影片的节目), 11.连载情况(影片的连载情况), 12.年龄分类(影片的年龄分类)。
** 目标是用最少的对话轮次确认用户的所有需求。提问要自然、有针对性,简洁直白**
不要一次性问太多问题,以下为你与用户对话历史记录{conversation_history}"""
    
    # 调用OpenAI API
    api_dou='4934c7fe-8359-4962-93fa-e80eb4e7f10a'
    dou_url='https://ark.cn-beijing.volces.com/api/v3'
    client = OpenAI(base_url=dou_url, api_key=api_dou)
    response = client.chat.completions.create(
        model="doubao-1-5-lite-32k-250115",
        messages=[
            {"role": "system", "content": "你是一个专业的客服助手，负责询问和理解用户的需求。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    # 获取生成的回答
    question = response.choices[0].message.content
    
    # 生成思考过程
    think_process = "根据对话历史分析用户可能的需求，生成针对性的问题以确认具体需求。"
    
    # 回答类型
    answer_type = "agent_question"
    
    return question, think_process, answer_type

def formulate_user_response(prompt: str) -> tuple:
    """
    生成用户的回答。
    
    Args:
        prompt (str): 用于生成用户回答的提示词
        
    Returns:
        tuple: (生成的回答, 回答类型, 思考过程)
    """
    # 调用OpenAI API
    api_dou='4934c7fe-8359-4962-93fa-e80eb4e7f10a'
    dou_url='https://ark.cn-beijing.volces.com/api/v3'
    client = OpenAI(base_url=dou_url, api_key=api_dou)
    response = client.chat.completions.create(
        model="doubao-1-5-lite-32k-250115",
        messages=[
            {"role": "system", "content": "你是一个用户，正在与客服对话。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    # 获取生成的回答
    answer = response.choices[0].message.content
    
    # 回答类型和思考过程
    answer_type = "user_response"
    think_process = "根据提供的需求生成自然的用户回答。"
    
    return answer, answer_type, think_process, None


import json

def calculate_metrics(data_file):
    # 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 过滤掉 completeness_rate 为 0 的记录
    filtered_data = [entry for entry in data if entry['completeness_rate'] != 0]

    # 打印被删除记录的数量
    deleted_count = len(data) - len(filtered_data)
    print(f"Deleted {deleted_count} entries with completeness_rate == 0.")

    # 如果需要保存修改后的数据到原文件或新文件
    output_file = "data/filtered_conversations1.json"  # 输出文件路径
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print(f"Filtered data has been saved to {output_file}.")


if __name__ == '__main__':
    data_file = "data/conversations.json"  # 数据文件路径
    calculate_metrics(data_file)