import json
import os


def calculate_brier_score(mastery_str, acc):
    """
    计算brier分数

    参数:
    mastery_str (str): 字符串形式的正确率，如 "76.03%"
    acc (float): 浮点形式的正确率，如 0.9

    返回:
    float: brier分数
    """
    # 将mastery字符串转换为0-1之间的小数
    mastery = float(mastery_str.strip('%')) / 100.0

    # 计算brier分数: (预测值 - 实际结果)^2
    # 这里假设实际结果是acc，预测值是mastery
    brier_score = (mastery - acc) ** 2

    return brier_score


def process_json_file(file_path):
    """
    处理JSON文件并计算每个条目的brier分数

    参数:
    file_path (str): JSON文件路径

    返回:
    list: 包含每个条目的brier分数的列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        brier_scores = []
        for item in data:
            mastery = item.get("1_mastery", "0%")
            acc = item.get("acc", 0.0)
            brier_score = calculate_brier_score(mastery, acc)
            brier_scores.append(brier_score)

        return brier_scores

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在。")
        return []
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 不是有效的JSON格式。")
        return []
    except Exception as e:
        print(f"错误：处理文件时发生意外错误: {e}")
        return []


def main():
    # 定义需要处理的文件路径列表
    file_paths = [
        "memory/exam_after_memory1.json",
        "memory/exam_after_memory1_AS.json",
        "memory/exam_after_memory2.json",
        "memory/exam_after_memory2_AS.json",
        "memory/exam_after_memory3.json",
        "memory/exam_after_memory3_AS.json"
    ]

    all_results = {}

    # 处理每个文件并收集结果
    for file_path in file_paths:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误：文件路径 '{file_path}' 不存在，跳过该文件。")
            continue

        # 处理文件并计算brier分数
        brier_scores = process_json_file(file_path)

        if brier_scores:
            # 计算平均brier分数
            average_brier = sum(brier_scores) / len(brier_scores)
            all_results[file_path] = {
                'count': len(brier_scores),
                'average_variance': 1 - average_brier
            }
        else:
            print(f"文件 '{file_path}' 未能计算任何brier分数。")

    # 输出所有结果
    print("\n==== 所有文件的统计结果 ====")
    for file_path, result in all_results.items():
        print(f"\n文件: {file_path}")
        print(f"  计算了 {result['count']} 个条目的brier分数。")
        print(f"  平均variance分数: {result['average_variance']:.4f}")

    # 这里预留位置供你添加写文件的代码
    # 你可以使用 all_results 字典中的数据进行文件写入操作


if __name__ == "__main__":
    main()