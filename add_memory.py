import json
import os
from math import exp


def update_memory_database():
    # 定义文件路径变量
    DIALOGUE_FILE = 'cosine/part8.json'
    INPUT_MEMORY_FILE = 'memory/memory_7.json'
    OUTPUT_MEMORY_FILE = 'memory/memory_8.json'

    # 读取对话集和记忆库
    try:
        with open(DIALOGUE_FILE, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到对话集文件 '{DIALOGUE_FILE}'")
        return
    except json.JSONDecodeError:
        print(f"错误：对话集文件格式不是有效的JSON")
        return

    try:
        with open(INPUT_MEMORY_FILE, 'r', encoding='utf-8') as f:
            memory = {item['id']: item for item in json.load(f)}
    except FileNotFoundError:
        print(f"错误：找不到记忆库文件 '{INPUT_MEMORY_FILE}'")
        return
    except json.JSONDecodeError:
        print(f"错误：记忆库文件格式不是有效的JSON")
        return

    # 遍历对话集
    for dialogue in dialogues:
        # 步骤1: 更新所有记忆项的time和mastery
        for item_id in memory:
            memory[item_id]['time'] += 1
            memory[item_id]['mastery'] *= exp(-0.005)

        # 步骤2: 更新相似项的mastery和time
        for i in range(1, 51):
            id_key = f"{i}_id"
            similarity_key = f"{i}_similarity"

            if id_key in dialogue and similarity_key in dialogue:
                related_id = dialogue[id_key]
                similarity = dialogue[similarity_key]

                if related_id in memory:
                    # 更新mastery
                    memory[related_id]['mastery'] += similarity
                    if memory[related_id]['mastery'] > 1:
                        memory[related_id]['mastery'] = 1

                    # 重置time
                    memory[related_id]['time'] = 0

    # 保存更新后的记忆库
    output_dir = os.path.dirname(OUTPUT_MEMORY_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(OUTPUT_MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(memory.values()), f, ensure_ascii=False, indent=2)

    print(f"成功更新并保存记忆库到 {OUTPUT_MEMORY_FILE}")


if __name__ == "__main__":
    update_memory_database()