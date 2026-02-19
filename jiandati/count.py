import json
import re
import os


def extract_scores(dialogue_text):
    """直接提取文本中第一组连续的5个指标评分，忽略题目分隔符"""
    # 匹配指标评分行（支持中文/英文冒号，允许分数后有空格或括号）
    pattern = re.compile(r'指标([一二三四五])[：:](\d+)(?:\s|\(.*?\))?')
    matches = pattern.findall(dialogue_text)

    if len(matches) < 5:
        return None  # 不足5个指标，直接返回

    # 查找连续的5个指标（一到五）
    index_map = {'一': 0, '二': 1, '三': 2, '四': 3, '五': 4}
    required_indices = set(range(5))

    # 遍历所有可能的连续5个匹配
    for i in range(len(matches) - 4):
        current_group = matches[i:i + 5]
        current_indices = set()
        score_list = [None] * 5

        for indicator, score in current_group:
            try:
                idx = index_map[indicator]
                score_val = int(score)
                if idx not in current_indices and 1 <= score_val <= 5:
                    current_indices.add(idx)
                    score_list[idx] = score_val
                else:
                    break  # 重复指标或分数无效，跳出循环
            except:
                break  # 转换失败，跳出循环
        else:  # 循环正常结束（未被break）
            if current_indices == required_indices:  # 包含所有五个指标
                return {
                    'score_1': score_list[0],
                    'score_2': score_list[1],
                    'score_3': score_list[2],
                    'score_4': score_list[3],
                    'score_5': score_list[4]
                }

    return None  # 未找到有效评分组

def process_json_file(input_path, output_path):
    # 以下代码与原脚本一致，省略重复部分（保持文件读写和错误处理逻辑）
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        error_items = []

        for index, item in enumerate(data):
            dialogue = item.get('dialogue', '')
            scores = extract_scores(dialogue)

            if scores:
                new_item = {**item, **scores}
                processed_data.append(new_item)
            else:
                error_items.append({
                    'index': index,
                    'dialogue': dialogue[:200] + '...' if len(dialogue) > 200 else dialogue
                })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        if error_items:
            print(f"发现 {len(error_items)} 个有问题的项:")
            for err in error_items:
                print(f"索引: {err['index']+1}")
                print(f"对话片段: {err['dialogue']}")
                print("-" * 50)

        print(f"处理完成！结果已保存到: {output_path}")

    except Exception as e:
        print(f"处理文件时出错: {e}")


if __name__ == "__main__":
    input_file = r"D:\python\memory\jiandati\score\111output_memory1.json"
    output_file = r"D:\python\memory\jiandati\score\score_111output_memory1.json"
    process_json_file(input_file, output_file)