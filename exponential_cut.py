import json
import os

def filter_and_save_conversations(input_filename, output_filename, zz):
    """
    从输入JSON文件读取对话数据，根据用户预期阈值过滤记录，截断对话，重新格式化记录，并将结果保存到新的JSON文件。

    参数:
        input_filename (str): 输入JSON文件路径(如"data.json")
        output_filename (str): 输出JSON文件路径(如"filtered_data.json")
        zz (float): 预期阈值 (termination_value)，用户对话的该值超过此阈值将触发记录的过滤和处理。
    """
    # --- 输入验证 ---
    # 检查输入文件是否存在
    if not os.path.exists(input_filename):
        print(f"错误: 输入文件 '{input_filename}' 未找到。")
        return
    # 检查阈值是否为非负数
    if not isinstance(zz, (int, float)) or zz < 0:
        print("错误: 阈值 'zz' 必须是一个非负数。")
        return

    try:
        # --- 读取输入JSON文件 ---
        with open(input_filename, 'r', encoding='utf-8') as f_in:
            # 加载JSON数据
            data = json.load(f_in)
        # 检查加载的数据是否为列表格式
        if not isinstance(data, list):
            print(f"错误: 输入文件 '{input_filename}' 应包含一个JSON列表。")
            return

    except json.JSONDecodeError:
        # 处理JSON解码错误
        print(f"错误: 无法从 '{input_filename}' 解码JSON。")
        return
    except Exception as e:
        # 处理其他文件读取错误
        print(f"读取输入文件时发生错误: {e}")
        return

    # --- 数据处理 ---
    filtered_data = [] # 初始化存储过滤后数据的列表
    # 遍历原始数据中的每一条记录
    for record in data:
        # 检查记录是否为字典且包含 'conversations' 键
        if not isinstance(record, dict) or "conversations" not in record:
            # 如果记录格式不正确，打印警告并跳过
            print(f"警告: 因缺少 'conversations' 跳过记录: {record.get('demand_index', 'N/A')}")
            continue

        conversations = record.get("conversations", []) # 安全地获取对话列表
        # 检查 'conversations' 是否为列表
        if not isinstance(conversations, list):
            # 如果 'conversations' 格式不正确，打印警告并跳过
            print(f"警告: 因 'conversations' 格式无效跳过记录: {record.get('demand_index', 'N/A')}")
            continue

        trigger_index = -1 # 初始化触发截断的用户对话索引，-1表示未找到

        # 查找第一个超过阈值的用户对话
        for i, turn in enumerate(conversations):
            # 检查当前轮次是否为有效字典
            if not isinstance(turn, dict):
                print(f"警告: 在记录 {record.get('demand_index', 'N/A')} 中跳过无效轮次: {turn}")
                continue # 跳过无效轮次

            # 检查是否为用户轮次
            if turn.get("role") == "user":
                # 获取用户的预期值 (termination_value)
                prospect = turn.get("termination_value")
                trigger_index = i
                # 检查预期值是否存在、是否为数字，并且是否大于阈值zz
                if prospect is not None and isinstance(prospect, (int, float)) and prospect < zz:
                    trigger_index = i # 记录当前用户对话的索引
                    break # 找到第一个满足条件的即停止查找

        # --- 记录处理和格式化 (仅当找到触发对话时) ---
        if trigger_index != -1:
            # 根据找到的索引截断对话列表 (包含触发对话本身)
            truncated_conversations = conversations[:trigger_index + 1]

            # --- 计算派生字段 ---

            # 1. 计算交互轮数 (用户发言次数)
            interaction_rounds = sum(1 for t in truncated_conversations if isinstance(t, dict) and t.get("role") == "user")

            # 2. 提取代理需求相关信息 (agent_demands)
            agent_keys = set() # 存储所有涉及的特征键 (使用集合自动去重)
            user_token_values = {} # 存储用户明确提供的特征值

            # 遍历截断后的对话
            for i, t in enumerate(truncated_conversations):
                if not isinstance(t, dict): continue # 再次检查，跳过无效轮次

                # --- 处理用户轮次 ---
                if t.get("role") == "user":
                    # 收集用户提及的特征 (target_feature)
                    target_features = t.get("target_feature")
                    if isinstance(target_features, list):
                        agent_keys.update(f for f in target_features if f is not None) # 添加列表中的非空特征
                    elif isinstance(target_features, str):
                        agent_keys.add(target_features) # 添加单个字符串特征

                    # 收集用户提供的键值对 (token_value)
                    feature_values = t.get("token_value")
                    if isinstance(feature_values, dict):
                        user_token_values.update(feature_values) # 更新用户提供的值
                        agent_keys.update(feature_values.keys()) # 添加这些值的键

                # --- 处理代理轮次 ---
                elif t.get("role") == "agent":
                    # 收集代理提及的特征 (token_value, 假设这里也可能包含特征名)
                    target_features_agent = t.get("token_value")
                    if isinstance(target_features_agent, list):
                        agent_keys.update(f for f in target_features_agent if f is not None)
                    elif isinstance(target_features_agent, str):
                        agent_keys.add(target_features_agent)

                    # 收集代理询问的特征 (target_feature)
                    agent_asked_features = t.get("target_feature")
                    if agent_asked_features is not None:
                        if isinstance(agent_asked_features, str):
                            agent_keys.add(agent_asked_features) # 添加单个询问的特征
                        elif isinstance(agent_asked_features, list):
                            agent_keys.update(f for f in agent_asked_features if f is not None) # 添加列表中询问的特征

                    # 收集代理提供的键值对 (feature_value) 的键
                    feature_values_agent = t.get("feature_value")
                    if isinstance(feature_values_agent, dict):
                        agent_keys.update(feature_values_agent.keys()) # 添加代理提供的值的键

                    # 特殊处理: 如果代理是基于配置文件的决策，则将代理提供的 feature_value 也视为用户的 token_value
                    if t.get('s_type') == "profile_based_determination":
                        profile_values = t.get("feature_value")
                        if isinstance(profile_values, dict):
                            # 仅当值非空时更新，避免覆盖用户明确提供的值（如果需要覆盖，则移除此检查）
                            if profile_values.values() is not None:
                                user_token_values.update(profile_values)


            # 清理可能存在的 None 值
            agent_keys.discard(None)

            # 格式化 agent_demands 字段
            agent_demands_formatted = {
                "key": sorted(list(agent_keys)), # 将收集到的键转换为排序列表，保证顺序一致性
                "token_value": user_token_values # 使用收集到的用户键值对
            }

            # --- 组装输出记录 ---
            output_record = {
                # 从原始记录中安全地获取字段值
                "demand_index": record.get("demand_index"),
                "conversations": truncated_conversations, # 使用截断后的对话
                "interaction_rounds": interaction_rounds, # 使用计算得到的交互轮数
                "matched_demand_count": record.get("matched_demand_count"),
                "completeness_rate": record.get("completeness_rate"),
                "redundancy_rate": record.get("redundancy_rate"),
                "agent_demands": agent_demands_formatted, # 使用格式化后的代理需求信息
                "original_user_need": record.get("original_user_need"), # 保留原始用户需求格式
                "user_type": record.get("user_type")
            }
            # 将处理好的记录添加到结果列表中
            filtered_data.append(output_record)

    # --- 写入输出JSON文件 ---
    try:
        with open(output_filename, 'w', encoding='utf-8') as f_out:
            # 将过滤和处理后的数据写入JSON文件
            # ensure_ascii=False 保证中文等非ASCII字符正确写入
            # indent=4 使输出的JSON文件格式化，易于阅读
            json.dump(filtered_data, f_out, ensure_ascii=False, indent=4)
        # 打印成功信息
        print(f"过滤后的数据已成功保存到 '{output_filename}'")
    except Exception as e:
        # 处理文件写入错误
        print(f"写入输出文件时发生错误: {e}")

# --- 配置 ---
INPUT_FILE = "data/filtered_conversations1.json" # 输入文件路径
OUTPUT_FILE = "expon0_85.json"                  # 输出文件路径
PROSPECT_THRESHOLD = 0.85                       # 设置用户预期阈值 (termination_value)

# --- 运行处理流程 ---
filter_and_save_conversations(INPUT_FILE, OUTPUT_FILE, PROSPECT_THRESHOLD)