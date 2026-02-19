import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def create_gradient_bar_chart(json_file_path, output_image_path):
    """
    从JSON文件读取数据并创建渐变色彩的柱状图

    参数:
    json_file_path (str): 输入JSON文件路径
    output_image_path (str): 输出图片路径
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取float字段的值
        float_values = [item['float'] for item in data]
        num_elements = len(float_values)

        # 确保有数据可处理
        if num_elements == 0:
            print("错误：JSON文件中没有数据")
            return

        # 创建自定义渐变颜色映射 - 从蓝到绿到黄
        colors = [(0, 0, 1), (0, 1, 0), (1, 1, 0)]  # 蓝 -> 绿 -> 黄
        cmap_name = 'blue_green_yellow'
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=1000)

        # 创建图表
        plt.figure(figsize=(15, 8))

        # 生成柱状图，每个柱子的颜色根据其值从颜色映射中获取
        x_positions = np.arange(num_elements)
        bars = plt.bar(x_positions, float_values, width=1.0,
                       color=cmap(float_values), edgecolor='none')

        # 设置图表标题和坐标轴标签
        plt.title(f'JSON数据可视化 ({num_elements}个数据点)')
        plt.xlabel('数据点索引')
        plt.ylabel('数值 (0-1范围)')

        # 设置y轴范围
        plt.ylim(0, 1)

        # 添加颜色条说明颜色与数值的对应关系
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('颜色渐变对应数值')

        # 调整布局
        plt.tight_layout()

        # 保存图表
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"图表已成功保存到: {output_image_path}")

        # 关闭图表
        plt.close()

    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file_path}")
    except json.JSONDecodeError:
        print(f"错误：文件 {json_file_path} 不是有效的JSON格式")
    except KeyError:
        print(f"错误：JSON对象中缺少'float'字段")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    # 输入JSON文件路径
    json_file = "result_pure/Deepseek-r1.json"
    # 输出图片路径
    output_image = "result_pure/picture/Deepseek-r1.png"

    # 执行绘图函数
    create_gradient_bar_chart(json_file, output_image)