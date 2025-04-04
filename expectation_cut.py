import json
import os

def filter_and_save_conversations(input_filename, output_filename, zz):
    """
    从输入JSON文件读取对话数据，根据用户预期阈值过滤记录，截断对话，重新格式化记录，并将结果保存到新的JSON文件。

    参数:
        input_filename (str): 输入JSON文件路径 (例如 "data.json")
        output_filename (str): 输出JSON文件路径 (例如 "filtered_data.json")
        zz (float): 预期阈值，用户对话的 termination_value 低于此值将触发过滤和截断
    """
    # --- 输入验证 ---
    # 检查输入文件是否存在
    if not os.path.exists(input_filename):
        print(f"错误: 输入文件 '{input_filename}' 未找到。")
        return
    # 检查阈值是否为非负数字
    if not isinstance(zz, (int, float)) or zz < 0:
        print("错误: 阈值 'zz' 必须是一个非负数字。")
        return

    try:
        # --- 读取输入JSON文件 ---
        with open(input_filename, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in) # 加载JSON数据
        # 检查输入数据是否为列表格式
        if not isinstance(data, list):
            print(f"错误: 输入文件 '{input_filename}' 应包含一个JSON列表。")
            return

    except json.JSONDecodeError:
        # 处理JSON解析错误
        print(f"错误: 无法从 '{input_filename}' 解析JSON。")
        return
    except Exception as e:
        # 处理其他读取文件时可能发生的错误
        print(f"读取输入文件时发生错误: {e}")
        return

    filtered_data = [] # 初始化用于存储过滤后数据的列表
    # --- 遍历原始数据中的每条记录 ---
    for record in data:
        # --- 记录结构验证 ---
        # 检查记录是否为字典且包含 'conversations' 键
        if not isinstance(record, dict) or "conversations" not in record:
            # demand_index 用于标识记录，如果不存在则显示 N/A
            print(f"警告: 因缺少 'conversations' 跳过记录: {record.get('demand_index', 'N/A')}")
            continue # 跳过当前记录

        conversations = record.get("conversations", []) # 获取对话列表，默认为空列表
        # 检查 'conversations' 是否为列表
        if not isinstance(conversations, list):
            print(f"警告: 因 'conversations' 格式无效跳过记录: {record.get('demand_index', 'N/A')}")
            continue # 跳过当前记录

        trigger_index = -1 # 初始化触发截断的用户对话索引，-1表示未找到

        # --- 查找低于阈值的用户对话 ---
        # 遍历对话中的每一轮 (turn)
        for i, turn in enumerate(conversations):
            # 检查当前轮次是否为字典结构
            if not isinstance(turn, dict):
                print(f"警告: 在记录 {record.get('demand_index', 'N/A')} 中跳过无效轮次: {turn}")
                continue # 跳过无效轮次

            # 检查是否是用户轮次
            if turn.get("role") == "user":
                prospect = turn.get("termination_value") # 获取用户的预期值
                trigger_index = i
                # 检查预期值是否存在、是数字且低于阈值 zz
                if prospect is not None and isinstance(prospect, (int, float)) and prospect > zz:
                    trigger_index = i # 记录下触发条件的轮次索引
                    break # 找到第一个满足条件的轮次后停止查找

        # --- 如果找到了触发轮次，则处理并格式化记录 ---
        if trigger_index != -1:
            # --- 截断对话 ---
            # 保留从开始到触发轮次（包含触发轮次）的所有对话
            truncated_conversations = conversations[:trigger_index + 1]

            # --- 计算派生字段 ---

            # 1. 计算交互轮数 (仅计算用户轮次)
            interaction_rounds = sum(1 for t in truncated_conversations if isinstance(t, dict) and t.get("role") == "user")

            # 2. 提取和格式化 agent_demands
            agent_keys = set() # 存储所有涉及的特征键 (使用set去重)
            user_token_values = {} # 存储用户提供的具体特征值

            # 遍历截断后的对话以收集信息
            for i, t in enumerate(truncated_conversations):
                if not isinstance(t, dict): continue # 跳过无效轮次

                # 从用户轮次收集信息
                if t.get("role") == "user":
                    # 收集用户提到的目标特征 (target_feature)
                    target_features = t.get("target_feature")
                    if isinstance(target_features, list):
                        # 如果是列表，添加所有非None的特征
                        agent_keys.update(f for f in target_features if f is not None)
                    elif isinstance(target_features, str):
                        # 如果是字符串，直接添加
                        agent_keys.add(target_features)

                    # 收集用户提供的特征值 (token_value)
                    feature_values = t.get("token_value")
                    if isinstance(feature_values, dict):
                        user_token_values.update(feature_values) # 更新用户提供的键值对
                        agent_keys.update(feature_values.keys()) # 将键也添加到 agent_keys

                # 从Agent轮次收集信息
                elif t.get("role") == "agent":
                    # 收集Agent提到的目标特征 (token_value，这里可能是指Agent询问的特征)
                    target_features = t.get("token_value")
                    if isinstance(target_features, list):
                        agent_keys.update(f for f in target_features if f is not None)
                    elif isinstance(target_features, str):
                        agent_keys.add(target_features)

                    # 收集Agent明确询问的特征 (target_feature)
                    feature_values = t.get("target_feature")
                    if feature_values is not None:
                        if isinstance(feature_values, str):
                            agent_keys.add(feature_values)
                        elif isinstance(feature_values, list):
                            agent_keys.update(feature_values)

                    # 特殊处理: 如果是基于配置文件的判断，提取 feature_value 作为用户提供的值
                    if t.get('s_type') == "profile_based_determination":
                        feature_values = t.get("feature_value")
                        if isinstance(feature_values, dict):
                            # 确保值不是None才更新 (尽管update会处理，这里更明确)
                            if feature_values.values() is not None:
                                user_token_values.update(feature_values)

            # 清理 agent_keys 中的 None 值
            agent_keys.discard(None)

            # 格式化 agent_demands
            agent_demands_formatted = {
                "key": sorted(list(agent_keys)), # 将特征键集合转为列表并排序，确保输出顺序一致
                "token_value": user_token_values # 包含用户提供的特征键值对
            }

            # --- 组装输出记录 ---
            output_record = {
                # 使用 .get() 安全地访问原始记录中的字段，避免因字段缺失而出错
                "demand_index": record.get("demand_index"),
                "conversations": truncated_conversations, # 截断后的对话
                "interaction_rounds": interaction_rounds, # 计算得到的交互轮数
                "matched_demand_count": record.get("matched_demand_count"), # 保留原始字段
                "completeness_rate": record.get("completeness_rate"), # 保留原始字段
                "redundancy_rate": record.get("redundancy_rate"), # 保留原始字段
                "agent_demands": agent_demands_formatted, # 格式化后的 agent 需求
                "original_user_need": record.get("original_user_need"), # 保留原始用户需求
                "user_type": record.get("user_type") # 保留用户类型
            }
            filtered_data.append(output_record) # 将处理后的记录添加到结果列表

    # --- 写入过滤后的数据到输出文件 ---
    try:
        with open(output_filename, 'w', encoding='utf-8') as f_out:
            # 将过滤后的数据写入JSON文件
            # ensure_ascii=False 保证中文等非ASCII字符正确显示
            # indent=4 使输出的JSON文件格式化，易于阅读
            json.dump(filtered_data, f_out, ensure_ascii=False, indent=4)
        print(f"过滤后的数据已成功保存到 '{output_filename}'")
    except Exception as e:
        # 处理写入文件时可能发生的错误
        print(f"写入输出文件时发生错误: {e}")

# --- 配置 ---
INPUT_FILE = "data/filtered_conversations1.json" # 输入文件名
OUTPUT_FILE = "except0_4.json"          # 输出文件名
PROSPECT_THRESHOLD = 0.4                  # 设置你期望的预期阈值

# --- 运行主流程 ---
filter_and_save_conversations(INPUT_FILE, OUTPUT_FILE, PROSPECT_THRESHOLD)