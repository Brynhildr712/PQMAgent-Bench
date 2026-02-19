import json
import numpy as np
from typing import List, Dict, Tuple, Optional


def load_data(file_path: str) -> List[Dict]:
    """从JSON文件加载数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
        return []
    except json.JSONDecodeError as e:
        print(f"错误: 文件 {file_path} 不是有效的JSON格式: {e}")
        return []


def parse_data(data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """解析数据，提取预测概率和实际准确率"""
    pred_probs = []
    accuracies = []

    for item in data:
        if '1_mastery' in item:
            mastery_str = item['1_mastery']
            try:
                pred_prob = float(mastery_str.strip('%')) / 100.0
                pred_probs.append(pred_prob)
            except ValueError:
                print(f"警告: 无法解析1_mastery值 '{mastery_str}'，跳过此样本")
                continue
        else:
            print("警告: 样本缺少'1_mastery'字段，跳过此样本")
            continue

        if 'acc' in item:
            acc = item['acc']
            if 0.0 <= acc <= 1.0:
                accuracies.append(acc)
            else:
                print(f"警告: acc值 '{acc}' 超出范围 [0, 1]，跳过此样本")
                continue
        else:
            print("警告: 样本缺少'acc'字段，跳过此样本")
            continue

    return np.array(pred_probs), np.array(accuracies)


def calculate_ece(pred_probs: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """计算预期校准误差(ECE)"""
    if len(pred_probs) != len(accuracies):
        raise ValueError("预测概率数组和实际准确率数组长度不一致")

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = np.logical_and(pred_probs >= bin_lower, pred_probs < bin_upper)

        if np.sum(in_bin) > 0:
            bin_confidence = np.mean(pred_probs[in_bin])
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_fraction = np.sum(in_bin) / len(pred_probs)

            ece += bin_fraction * np.abs(bin_confidence - bin_accuracy)

    return ece


def process_file(file_path: str) -> Optional[float]:
    """处理单个文件并返回ECE值"""
    print(f"\n正在处理文件: {file_path}")
    data = load_data(file_path)
    if not data:
        print("没有数据可处理")
        return None

    pred_probs, accuracies = parse_data(data)

    if len(pred_probs) == 0:
        print("解析后没有有效数据")
        return None

    print(f"成功解析 {len(pred_probs)} 个有效样本")
    return calculate_ece(pred_probs, accuracies)


def evaluate_ece(ece: float) -> str:
    """根据ECE值评估校准质量"""
    if ece < 0.05:
        return "校准良好"
    elif 0.05 <= ece < 0.1:
        return "校准一般"
    else:
        return "校准较差"


def main():
    # 定义要处理的文件列表
    files_to_process = [
        "memory/exam_after_memory1.json",
        "memory/exam_after_memory1_AS.json",
        "memory/exam_after_memory2.json",
        "memory/exam_after_memory2_AS.json",
        "memory/exam_after_memory3.json",
        "memory/exam_after_memory3_AS.json"
    ]

    # 存储每个文件的结果
    results = {}

    # 处理每个文件
    for file_path in files_to_process:
        ece = process_file(file_path)
        if ece is not None:
            results[file_path] = ece

    # 输出汇总结果
    print("\n===== 汇总结果 =====")
    if results:
        for file_path, ece in results.items():
            print(f"{file_path}: ECE = {(4*ece):.4f}")

        # 计算平均ECE
        avg_ece = sum(results.values()) / len(results)
        print(f"\n所有文件的平均ECE: {avg_ece:.4f}")
    else:
        print("没有计算任何ECE值")


if __name__ == "__main__":
    main()

"""
    计算预期校准误差(ECE)

    参数:
    pred_probs: 预测概率数组
    accuracies: 实际准确率数组
    num_bins: 分箱数量

    返回:
    ece: 预期校准误差

    ECE计算方式说明:
    1. 将预测概率范围[0,1]均分为num_bins个区间(分箱)
    2. 对于每个分箱:
       - 计算该分箱内样本的平均预测概率(confidence)
       - 计算该分箱内样本的实际平均准确率(accuracy)
       - 计算该分箱的样本占比(fraction)
       - 计算该分箱的校准误差: |confidence - accuracy|
    3. ECE = Σ (fraction_i * |confidence_i - accuracy_i|) 对所有分箱i求和

    ECE标准说明:
    - ECE值范围为[0,1]，值越小表示模型校准越好
    - 完美校准的模型ECE=0(所有分箱的预测概率等于实际准确率)
    - 一般来说:
      - ECE<0.05: 校准良好
      - 0.05≤ECE<0.1: 校准一般
      - ECE≥0.1: 校准较差
"""