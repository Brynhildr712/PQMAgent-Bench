import json
import os


def calculate_brier_score(mastery_str, acc):
    """
    计算Brier分数

    参数:
    mastery_str (str): 字符串形式的正确率预测，如 "76.03%"
    acc (float): 实际正确率，如 0.5 表示答对5题，答错5题

    返回:
    float: Brier分数
    """
    # 将mastery字符串转换为0-1之间的小数
    p = float(mastery_str.strip('%')) / 100.0

    # 计算答对题数和答错题数（假设共10题）
    correct = int(round(acc * 10))
    incorrect = 10 - correct

    # 答对题的Brier分数: (p-1)^2，共correct题
    # 答错题的Brier分数: (p-0)^2，共incorrect题
    # 总Brier分数: [correct*(p-1)^2 + incorrect*p^2] / 10
    brier_score = (correct * (p - 1) ** 2 + incorrect * p ** 2) / 10

    return brier_score


def process_json_file(file_path):
    """
    处理JSON文件并计算每个条目的Brier分数

    参数:
    file_path (str): JSON文件路径

    返回:
    list: 包含每个条目的Brier分数的列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        brier_scores = []
        for item in data:
            mastery = item.get("1_mastery", "0%")
            acc = item.get("acc", 0.0)

            # 验证acc是否在有效范围内
            if not (0 <= acc <= 1):
                print(f"警告: 条目 {item} 的acc值 {acc} 超出范围 [0, 1]，将跳过此条目")
                continue

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
    # 定义要处理的文件列表，可随时添加或删除文件
    file_paths = [
        "memory/exam_after_memory1.json",
        "memory/exam_after_memory1_AS.json",
        "memory/exam_after_memory2.json",
        "memory/exam_after_memory2_AS.json",
        "memory/exam_after_memory3.json",
        "memory/exam_after_memory3_AS.json"
    ]

    all_brier_scores = []

    # 处理每个文件
    for file_path in file_paths:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误：文件路径 '{file_path}' 不存在，将跳过此文件。")
            continue

        # 处理文件并计算Brier分数
        brier_scores = process_json_file(file_path)
        all_brier_scores.extend(brier_scores)

        if brier_scores:
            # 计算当前文件的平均Brier分数
            average_brier = sum(brier_scores) / len(brier_scores)
            print(f"文件 '{file_path}' 的统计结果:")
            print(f"  计算了 {len(brier_scores)} 个条目的Brier分数。")
            print(f"  平均Brier分数: {(1-average_brier):.4f}")
        else:
            print(f"文件 '{file_path}' 未能计算任何Brier分数。")

    # 计算所有文件的总体统计结果
    if all_brier_scores:
        overall_average = sum(all_brier_scores) / len(all_brier_scores)
        print("\n所有文件的总体统计结果:")
        print(f"  共计算了 {len(all_brier_scores)} 个条目的Brier分数。")
        print(f"  总体平均Brier分数: {overall_average:.4f}")
    else:
        print("\n未能从任何文件中计算Brier分数。")


if __name__ == "__main__":
    main()