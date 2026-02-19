import json
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_json_file(file_path):
    """加载JSON文件并返回数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。")
        return []
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSON格式。")
        return []


def save_to_json(data, file_path):
    """将数据保存到JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_top_similar_items(query_embedding, knowledge_embeddings, knowledge_data, top_n=11):  # 修改1：top_n从50改为10
    """获取最相似的前N个知识项"""
    # 将张量从GPU移到CPU并转换为NumPy数组
    if query_embedding.is_cuda:
        query_embedding = query_embedding.cpu().numpy()
    if knowledge_embeddings.is_cuda:
        knowledge_embeddings = knowledge_embeddings.cpu().numpy()

    # 计算余弦相似度
    similarities = cosine_similarity([query_embedding], knowledge_embeddings)[0]

    # 获取前N个相似项的索引（降序排列）
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    # 提取对应的知识项
    top_items = []
    for idx in top_indices:
        item = {
            'id': knowledge_data[idx]['id'],
            'similarity': float(similarities[idx]),  # 转换为Python原生float
            'content': knowledge_data[idx]['knowledge']  # 新增：保存匹配的知识库内容
        }
        top_items.append(item)

    return top_items


def process_files(dialogue_file, knowledge_file, output_file, model_path):
    """处理文件并生成结果"""
    dialogue_data = load_json_file(dialogue_file)
    knowledge_data = load_json_file(knowledge_file)

    if not dialogue_data or not knowledge_data:
        print("数据加载失败，程序退出。")
        return

    # 加载本地模型
    print(f"正在从 {model_path} 加载模型...")
    model = SentenceTransformer(model_path)

    # 编码知识库
    print("正在编码知识库...")
    knowledge_texts = [item['knowledge'] for item in knowledge_data]
    knowledge_embeddings = model.encode(knowledge_texts, convert_to_tensor=True)

    # 处理对话数据
    print("正在处理对话数据...")
    total_items = len(dialogue_data)
    processed_items = 0
    start_time = time.time()

    for item in dialogue_data:
        # 拼接d-point和d-content
        query_text = f"{item['points']} {item['content']}"

        # 编码查询文本
        query_embedding = model.encode(query_text, convert_to_tensor=True)

        # 获取最相似的10个知识项（修改1影响此处）
        top_items = get_top_similar_items(query_embedding, knowledge_embeddings, knowledge_data)

        # 添加新字段 - 最相似的10个（修改1影响此处）
        for i, top_item in enumerate(top_items, 1):
            item[f"{i}_id"] = top_item['id']
            item[f"{i}_similarity"] = top_item['similarity']
            item[f"{i}_content"] = top_item['content']  # 新增：保存知识库内容

        # 进度更新
        processed_items += 1
        if processed_items % 10 == 0:
            elapsed_time = time.time() - start_time
            items_per_second = processed_items / elapsed_time
            remaining_items = total_items - processed_items
            estimated_remaining_time = remaining_items / items_per_second if items_per_second > 0 else 0

            print(f"进度: {processed_items}/{total_items} ({processed_items / total_items * 100:.2f}%) | "
                  f"速度: {items_per_second:.2f}项/秒 | "
                  f"预计剩余时间: {estimated_remaining_time:.2f}秒")

    # 保存结果
    print("正在保存结果...")
    save_to_json(dialogue_data, output_file)
    print(f"处理完成！结果已保存到 {output_file}")


if __name__ == "__main__":
    # 文件路径
    dialogue_file = "data/knowledge_only.json"
    knowledge_file = "data/knowledge_only.json"
    output_file = "data/cosine_top10.json"  # 修改1影响此处：文件名从top50改为top10

    # 本地模型路径
    model_path = r"D:\python\memory\all-MiniLM-L12-v2"  # 修改为您的实际路径

    # 检查文件是否存在
    if not os.path.exists(dialogue_file):
        print(f"错误：对话文件 {dialogue_file} 不存在。")
    elif not os.path.exists(knowledge_file):
        print(f"错误：知识文件 {knowledge_file} 不存在。")
    elif not os.path.exists(model_path):
        print(f"错误：模型路径 {model_path} 不存在。请先下载模型。")
    else:
        # 处理文件
        process_files(dialogue_file, knowledge_file, output_file, model_path)