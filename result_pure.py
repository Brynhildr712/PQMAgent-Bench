import json
import os


def sort_json_by_float(input_file, output_file):
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 按float字段排序
        sorted_data = sorted(data, key=lambda x: x['float'])

        # 写入排序后的JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, ensure_ascii=False, indent=2)

        print(f"已成功排序并保存到 {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except json.JSONDecodeError:
        print(f"错误：无法解析 {input_file} 中的JSON内容")
    except KeyError:
        print(f"错误：JSON元素中缺少 'float' 字段")
    except Exception as e:
        print(f"发生未知错误：{e}")


if __name__ == "__main__":
    input_file = "result_pure/Deepseek-v3.json"
    output_file = "result_pure_smooth/Deepseek-v3.json"
    sort_json_by_float(input_file, output_file)

    """
        file_paths = [
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
    ]"""