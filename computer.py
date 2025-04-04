import re

from Levenshtein import ratio
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
model_path="D:/pyproject/bert_base_chinese"
def get_word_embedding(word_list):
    """
    获取词语列表中每个词语的BERT嵌入向量（去除[CLS]和[SEP]后的平均池化）

    Args:
        word_list: 词语列表

    Returns:
        torch.Tensor: 词语嵌入向量，维度为 [len(word_list), hidden_size]
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    # 批量编码
    encoded = tokenizer(word_list, padding=True, truncation=True, return_tensors='pt')

    # 获取BERT输出
    with torch.no_grad():
        outputs = model(**encoded)

    hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
    attention_mask = encoded['attention_mask']

    # 去除每个序列的[CLS]和[SEP]标记
    hidden_states = hidden_states[:, 1:-1, :]  # 切片去除首尾标记
    attention_mask = attention_mask[:, 1:-1]   # 相应地调整attention mask

    # 计算平均值（考虑attention mask）
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    hidden_states_masked = hidden_states * mask_expanded
    sum_embeddings = torch.sum(hidden_states_masked, dim=1)  # 在序列长度维度求和
    sum_mask = torch.clamp(attention_mask.sum(dim=1).unsqueeze(-1), min=1e-9)  # 防止除零
    mean_embeddings = sum_embeddings / sum_mask

    return mean_embeddings

def calculate_similarity_matrix(list1, list2):
    """
    计算两个词语列表之间的余弦相似度矩阵

    Args:
        list1: 第一个词语列表
        list2: 第二个词语列表

    Returns:
        numpy.ndarray: 相似度矩阵，维度为 [len(list1), len(list2)]
    """
    # 获取两个列表的词向量
    embeddings1 = get_word_embedding(list1)
    embeddings2 = get_word_embedding(list2)

    # 归一化向量
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    # 计算余弦相似度矩阵
    similarity_matrix = torch.mm(embeddings1, embeddings2.transpose(0, 1))
    return similarity_matrix.numpy()


def dsc_bert(prediction, gold, threshold):
    """
    使用BERT计算两个列表的Dice Similarity Coefficient (DSC)

    Args:
        prediction: 预测的词语列表
        gold: 真实的词语列表
        threshold: 相似度阈值

    Returns:
        float: DSC值
    """
    similarity_matrix = calculate_similarity_matrix(prediction, gold)
    hit_p = 0
    for i, _ in enumerate(prediction):
        for j, _ in enumerate(gold):
            if similarity_matrix[i, j] >= threshold:
                hit_p += 1
    return (2 * hit_p) / (len(prediction) + len(gold)) if (len(prediction) + len(gold)) > 0 else 0.0

def dsc_levenshtein(prediction, gold, threshold):
    """
    使用Levenshtein距离计算两个列表的Dice Similarity Coefficient (DSC)

    Args:
        prediction: 预测的词语列表
        gold: 真实的词语列表
        threshold: 相似度阈值

    Returns:
        float: DSC值
    """
    hit_p = 0
    for p in prediction:
        for g in gold:
            if ratio(p, g) >= threshold:
                hit_p += 1
    return (2 * hit_p) / (len(prediction) + len(gold)) if (len(prediction) + len(gold)) > 0 else 0.0


def jaccard_levenshtein(prediction, gold, threshold):
    """
    使用Levenshtein距离计算两个列表的Jaccard相似度

    Args:
        prediction: 预测的词语列表
        gold: 真实的词语列表
        threshold: 相似度阈值

    Returns:
        float: Jaccard相似度
    """
    hit_p = 0
    for p in prediction:
        for g in gold:
            if ratio(p, g) >= threshold:
                hit_p += 1
    return hit_p / (len(prediction) - hit_p + len(gold)) if (len(prediction) - hit_p + len(gold)) > 0 else 0.0

def jaccard_bert(prediction, gold, threshold):
    """
    使用BERT计算两个列表的Jaccard相似度

    Args:
        prediction: 预测的词语列表
        gold: 真实的词语列表
        threshold: 相似度阈值

    Returns:
        float: Jaccard相似度
    """
    similarity_matrix = calculate_similarity_matrix(prediction, gold)
    hit_p = 0
    for i, _ in enumerate(prediction):
        for j, _ in enumerate(gold):
            if similarity_matrix[i, j] >= threshold:
                hit_p += 1
    return hit_p / (len(prediction) - hit_p + len(gold)) if (len(prediction) - hit_p + len(gold)) > 0 else 0.0


def calculate_metrics(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_completeness_rate = 0
    total_features_hit_rate = 0
    total_set_similarity_levenshtein = 0
    total_set_similarity_bert = 0
    num_entries = len(data)
    z=0
    for entry in data:
        print('进度',z)
        z+=1
        # 1. Features命中率：original_user_need[keys]中的特征在agent_demands[key]中的比例
        original_keys = entry['original_user_need']['keys']
        agent_keys = entry['agent_demands']['key']
        hit_count = sum(1 for key in original_keys if key in agent_keys)
        features_hit_rate = hit_count / len(agent_keys) if len(agent_keys) > 0 else 0.0
        total_features_hit_rate += features_hit_rate

        # 2. 信息完整度：agent_demands[token_value]与original_user_need[token-values]相同值的比例
        original_values = entry['original_user_need']['token-values']
        agent_values = entry['agent_demands']['token_value']
        if len(agent_values) != 0:
            matched_values = sum(1 for key in original_values if key in agent_values and agent_values[key] == original_values[key])
            completeness_rate = matched_values / len(original_values) if len(original_values) > 0 else 0.0
            total_completeness_rate += completeness_rate

            # 3. 集合相似度：比较两个token_value的keys+values列表的相似度
            agent_list = list(agent_values.keys()) + list(agent_values.values())
            original_list = list(original_values.keys()) + list(original_values.values())
            # 过滤掉None和空字符串
            agent_list = [str(item) for item in agent_list if item is not None and str(item).strip() != ""]
            original_list = [str(item) for item in original_list if item is not None and str(item).strip() != ""]

            total_set_similarity_levenshtein += jaccard_levenshtein(agent_list, original_list, 0.8)
            total_set_similarity_bert += jaccard_bert(agent_list, original_list, 0.8)

    avg_completeness_rate = total_completeness_rate / num_entries
    avg_features_hit_rate = total_features_hit_rate / num_entries
    avg_set_similarity_levenshtein = total_set_similarity_levenshtein / num_entries
    avg_set_similarity_bert = total_set_similarity_bert / num_entries

    print(f"平均信息完整度: {avg_completeness_rate:.4f}")
    print(f"平均Features命中率: {avg_features_hit_rate:.4f}")
    print(f"平均集合相似度 (Levenshtein): {avg_set_similarity_levenshtein:.4f}")
    print(f"平均集合相似度 (BERT): {avg_set_similarity_bert:.4f}")


if __name__ == '__main__':
    data_file = "normalcut1.json"  # 数据文件路径
    calculate_metrics(data_file)