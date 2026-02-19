import json
from collections import defaultdict


def analyze_probability_differences(file_paths):
    # 存储所有文件的统计结果
    all_results = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # 初始化计数器
            counts = defaultdict(int)
            total = 0

            # 处理每个数据项
            for item in data:
                # 提取预测概率并转换为小数
                mastery_str = item.get('1_mastery', '0%').strip('%')
                predicted = float(mastery_str) / 100.0

                # 提取实际概率
                actual = float(item.get('acc', 0))

                # 计算差异百分比（实际 - 预测）
                diff_percentage = (actual - predicted) * 100

                # 分类计数
                if diff_percentage > 25:
                    counts['实际比预测高20%以上'] += 1
                elif -25 <= diff_percentage <= 25:
                    counts['实际比预测低20%~高20%'] += 1
                else:
                    counts['实际比预测低20%以上'] += 1

                total += 1

            # 计算占比
            result = {
                'file': file_path,
                '实际比预测高20%以上': counts['实际比预测高20%以上'] / total * 100 if total > 0 else 0,
                '实际比预测低20%~高20%': counts['实际比预测低20%~高20%'] / total * 100 if total > 0 else 0,
                '实际比预测低20%以上': counts['实际比预测低20%以上'] / total * 100 if total > 0 else 0
            }

            all_results.append(result)

        except FileNotFoundError:
            print(f"错误: 文件 {file_path} 不存在")
        except json.JSONDecodeError:
            print(f"错误: 文件 {file_path} 不是有效的JSON格式")
        except Exception as e:
            print(f"处理文件 {file_path} 时发生未知错误: {e}")

    # 输出表格
    if all_results:
        print(
            f"{'文件名':<30} {'实际比预测高20%以上(%)':<20} {'实际比预测低20%~高20%(%)':<20} {'实际比预测低20%以上(%)':<20}")
        print("-" * 90)
        for result in all_results:
            print(
                f"{result['file']:<60} {result['实际比预测高20%以上']:>10.0f} {result['实际比预测低20%~高20%']:>10.0f} {result['实际比预测低20%以上']:>10.0f}{result['实际比预测高20%以上']+result['实际比预测低20%~高20%']+result['实际比预测低20%以上']:>10.2f}")


# 使用示例
if __name__ == "__main__":
    # 请将下面的文件名替换为你的实际JSON文件路径
    file_names = [
        "memory/exam_after_memory1.json",
        "memory/exam_after_memory1_AS.json",
        "memory/exam_after_memory1_AS2.json",
        "memory/exam_after_memory2.json",
        "memory/exam_after_memory2_AS.json",
        "memory/exam_after_memory2_AS2.json",
        "memory/exam_after_memory3.json",
        "memory/exam_after_memory3_AS.json",
        "memory/exam_after_memory3_AS2.json",
        "multi_exam/multi_exam_after_memory1.json",
        "multi_exam/multi_exam_after_memory2.json",
        "multi_exam/multi_exam_after_memory3.json",
        "other_method_RoleLLM/RoleLLM_exam_after_memory.json",
        "other_method_RoleLLM/RoleLLM_multi_exam_after_memory.json",
        "other_method_TIM/TiM_exam_after_memory.json",
        "other_method_TIM/TiM_multi_exam_after_memory.json",
        "other_method_MemoryBank/MB_exam_after_memory.json",
        "other_method_MemoryBank/MB_multi_exam_after_memory.json",
        "other_method_CLLM/CLLM_exam_after_memory.json",
        "other_method_CLLM/CLLM_multi_exam_after_memory.json",
        "other_model_answer/exam_after_claude-3.7.json",
        "other_model_answer/exam_after_Deepseek-r1.json",
        "other_model_answer/exam_after_gemini-2.5-pro.json",
        "other_model_answer/exam_after_glm-4.json",
        "other_model_answer/exam_after_gpt-4.json",
        "other_model_answer/exam_after_gpt-4o.json",
        "other_model_answer/exam_after_Qwen2.5-max.json",
        "other_model_answer/exam_after_Qwen3-plus.json"
    ]
    analyze_probability_differences(file_names)