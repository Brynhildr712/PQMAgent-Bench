import json
import os
from typing import Dict, Any, List


def score_to_coefficient(score: str) -> float:
    """将d-score转换为对应的系数"""
    score_map = {
        "熟练掌握": 1.0,
        "基本掌握": 0.7,
        "部分掌握": 0.4,
        "未掌握": 0.0
    }
    return score_map.get(score, 0.0)


def process_similarity_data(input_path: str, output_path: str) -> None:
    """处理相似度数据并保存到新文件"""
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 读取原始JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 处理每个元素
        for item in data:
            # 获取系数
            score = item.get("d-score", "")
            coefficient = score_to_coefficient(score)

            # 处理50个相似度值（1st到50th）
            for i in range(1, 51):
                key = f"{i}_similarity"
                if key in item:
                    # 应用系数
                    item[key] *= coefficient
                    # 应用0.95的幂次（第n个相似度乘以0.95^(n-1)）
                    item[key] *= (0.95 ** (i - 1))
                    # 保留4位小数
                    item[key] = round(item[key], 4)

        # 保存处理后的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"处理完成，结果已保存到 {output_path}")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_path}")
    except json.JSONDecodeError:
        print(f"错误：输入文件 {input_path} 不是有效的JSON格式")
    except Exception as e:
        print(f"发生未知错误：{e}")


if __name__ == "__main__":
    INPUT_PATH = "data/cosine_top50.json"  # 请确保输入文件路径正确
    OUTPUT_PATH = "data/real_cosine_top50.json"
    process_similarity_data(INPUT_PATH, OUTPUT_PATH)