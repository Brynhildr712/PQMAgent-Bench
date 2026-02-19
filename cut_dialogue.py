import json
import re


def process_dialogue(dialogue):
    """处理dialogue字段，提取知识点三元组"""
    pattern = r'知识点：(.*?)\s+具体内容：(.*?)\s+学生掌握程度：(.*?)(?=\n\n|$)'
    matches = re.findall(pattern, dialogue, re.DOTALL)
    return [
        {
            "d-point": match[0].strip(),
            "d-content": match[1].strip(),
            "d-score": match[2].strip()
        }
        for match in matches
    ]


def main():
    try:
        # 读取原始JSON文件
        with open('dialogue_all_with_id.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        new_data = []
        missing_triples_indices = []

        # 处理每个元素
        for i, item in enumerate(data):
            dialogue = item.get('dialogue', '')
            triples = process_dialogue(dialogue)

            if not triples:
                missing_triples_indices.append(i)
                print(f"\n警告：第 {i} 组元素的dialogue字段未找到三元组")

                # 打印points和dialogue字段内容
                points = item.get('points', '')
                print(f"  points字段内容: {points}")
                print(f"  dialogue字段内容: {dialogue[:300]}...")  # 限制长度避免输出过长

            # 为每个三元组创建新元素
            for triple in triples:
                new_item = {
                    "d-point": triple["d-point"],
                    "d-content": triple["d-content"],
                    "d-score": triple["d-score"]
                }

                # 保留原JSON元素的其他字段
                for field in ["id", "points", "content"]:
                    if field in item:
                        new_item[field] = item[field]

                new_data.append(new_item)

        # 写入新JSON文件
        with open('new_dialogue.json', 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

        print(f"\n处理完成，共生成 {len(new_data)} 个三元组")
        if missing_triples_indices:
            print(f"共有 {len(missing_triples_indices)} 组元素缺少三元组，序号分别为：{missing_triples_indices}")

    except FileNotFoundError:
        print("错误：找不到dialogue_all.json文件")
    except json.JSONDecodeError:
        print("错误：JSON格式解析错误")
    except Exception as e:
        print(f"发生未知错误：{e}")


if __name__ == "__main__":
    main()