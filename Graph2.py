import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


# 自动检测系统中可用的中文字体
def get_available_chinese_fonts():
    chinese_fonts = []
    for font in fm.findSystemFonts():
        try:
            font_prop = fm.FontProperties(fname=font)
            if font_prop.get_name() and (
                    'hei' in font_prop.get_name().lower() or
                    'song' in font_prop.get_name().lower() or
                    'kai' in font_prop.get_name().lower() or
                    'fang' in font_prop.get_name().lower() or
                    'microsoft' in font_prop.get_name().lower()
            ):
                chinese_fonts.append(font_prop.get_name())
        except:
            continue
    return list(set(chinese_fonts))


# 设置中文字体
available_fonts = get_available_chinese_fonts()
if available_fonts:
    plt.rcParams["font.family"] = available_fonts[0]
    print(f"使用可用的中文字体: {available_fonts[0]}")
else:
    print("未找到可用的中文字体，图表中的中文可能无法正常显示。")
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def smooth_data(data, window_size=10):
    """
    对数据进行移动平均平滑处理
    :param data: 原始数据（一维数组）
    :param window_size: 滑动窗口大小（需为正整数，越大平滑效果越强）
    :return: 平滑后的数据（与原始数据长度相同）
    """
    if len(data) <= window_size or window_size < 2:
        return data  # 数据量小于窗口大小时不处理

    # 确保窗口大小为奇数（减少边缘偏差）
    if window_size % 2 == 0:
        window_size += 1

    # 生成滑动窗口（权重平均，窗口内每个值的权重相同）
    window = np.ones(window_size) / window_size

    # 使用卷积实现移动平均（保持与原始数据长度一致）
    smoothed_data = np.convolve(data, window, mode='same')

    # 处理边缘（避免窗口超出边界导致的偏差）
    smoothed_data[:window_size // 2] = data[:window_size // 2]  # 前半部分保留原始数据
    smoothed_data[-window_size // 2:] = data[-window_size // 2:]  # 后半部分保留原始数据

    return smoothed_data


def load_data(file_path):
    """加载单个JSON文件中的数据，计算绝对偏差值"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            abs_errors = []

            for item in data:
                try:
                    predicted = float(item.get('1_mastery', '0%').strip('%')) / 100.0
                    actual = float(item.get('acc', 0))
                    abs_error = abs(actual - predicted)
                    abs_errors.append(abs_error)
                except (ValueError, TypeError):
                    continue

            if abs_errors:
                print(f"从 {file_path} 加载了 {len(abs_errors)} 个数据点")
                return np.array(abs_errors)
            else:
                print(f"文件 {file_path} 不包含有效数据")
                return np.array([])

    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return np.array([])


def create_single_error_chart(abs_errors, file_name, output_path=None, smooth_window=10):
    """为单个文件创建平滑后的绝对偏差值分布图表"""
    if not len(abs_errors):
        print(f"文件 {file_name} 没有有效数据用于绘图")
        return None

    # 对绝对偏差值进行排序（从小到大）
    sorted_errors = np.sort(abs_errors)

    # 平滑处理
    smoothed_errors = smooth_data(sorted_errors, window_size=smooth_window)

    n_points = len(smoothed_errors)
    x = np.arange(1, n_points + 1)  # 横坐标：数据点索引

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 绘制平滑后的曲线
    plt.plot(x, smoothed_errors, color='#3498db', linewidth=2, alpha=0.8,
             label=f'平滑曲线（窗口大小={smooth_window}）')

    # 可选：绘制原始曲线（浅色作为对比）
    plt.plot(x, sorted_errors, color='#95a5a6', linewidth=1, alpha=0.3, label='原始曲线')

    # 添加参考线（y=0）
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 添加误差分布的统计信息
    mean_error = np.mean(sorted_errors)
    median_error = np.median(sorted_errors)
    std_error = np.std(sorted_errors)
    max_error = np.max(sorted_errors)
    min_error = np.min(sorted_errors)

    plt.figtext(0.15, 0.85,
                f'统计信息:\n'
                f'• 数据点总数: {n_points:,}\n'
                f'• 平均绝对偏差: {mean_error:.4f}\n'
                f'• 中位数绝对偏差: {median_error:.4f}\n'
                f'• 绝对偏差标准差: {std_error:.4f}\n'
                f'• 最大绝对偏差: {max_error:.4f}\n'
                f'• 最小绝对偏差: {min_error:.4f}',
                bbox=dict(facecolor='wheat', alpha=0.5), fontsize=10)

    # 设置图表标题和坐标轴标签
    plt.title(f'{file_name} - 绝对偏差值分布（从小到大排序，平滑处理）', fontsize=16)
    plt.xlabel('数据点索引（按绝对偏差值从小到大排序）', fontsize=12)
    plt.ylabel('绝对偏差值 |实际 - 预测|', fontsize=12)

    # 设置坐标轴范围和刻度
    plt.grid(True, linestyle='--', alpha=0.7)
    y_max = max(smoothed_errors) * 1.1
    plt.ylim(0, y_max)

    # 添加图例
    plt.legend(loc='best')

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {output_path}")

    plt.close()
    return output_path


def create_comparison_chart(all_abs_errors, file_names, output_path=None, smooth_window=10):
    """创建平滑后的比较图表"""
    # 过滤掉空数据
    valid_indices = [i for i, errors in enumerate(all_abs_errors) if len(errors) > 0]
    if not valid_indices:
        print("没有有效数据用于创建比较图表")
        return None

    valid_errors = [all_abs_errors[i] for i in valid_indices]
    valid_names = [file_names[i] for i in valid_indices]

    # 创建图表
    plt.figure(figsize=(14, 10))

    # 生成颜色列表
    colors = plt.cm.tab20(np.linspace(0, 1, len(valid_names)))

    # 绘制每个文件的平滑曲线
    for i, (errors, name, color) in enumerate(zip(valid_errors, valid_names, colors)):
        # 排序并平滑
        sorted_errors = np.sort(errors)
        smoothed_errors = smooth_data(sorted_errors, window_size=smooth_window)

        n_points = len(smoothed_errors)
        x = np.arange(1, n_points + 1)

        # 绘制平滑曲线
        plt.plot(x, smoothed_errors, color=color, linewidth=2, alpha=0.8, label=name)

    # 添加参考线（y=0）
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1.5)

    # 设置图表标题和坐标轴标签
    plt.title(f'所有文件的绝对偏差值分布比较（从小到大排序，平滑窗口={smooth_window}）', fontsize=16)
    plt.xlabel('数据点索引（按绝对偏差值从小到大排序）', fontsize=12)
    plt.ylabel('绝对偏差值 |实际 - 预测|', fontsize=12)

    # 设置坐标轴范围和刻度
    plt.grid(True, linestyle='--', alpha=0.7)
    all_max_errors = [np.max(errors) for errors in valid_errors]
    y_max = max(all_max_errors) * 1.1
    plt.ylim(0, y_max)

    # 添加图例
    plt.legend(title="数据文件", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # 保存图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"比较图表已保存至: {output_path}")

    plt.close()
    return output_path


def main():
    """主函数"""
    file_paths = [
        "memory/exam_after_memory1.json",
        "memory/exam_after_memory1_AS.json",
        "memory/exam_after_memory1_AS2.json"
    ]

    output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "概率分布图")
    os.makedirs(output_dir, exist_ok=True)

    print("开始处理数据...")

    all_abs_errors = []
    file_names = []

    # 处理每个文件（可调整平滑窗口大小，建议5-20之间）
    smooth_window = 5  # 窗口越大，平滑效果越强（根据1400个数据点，10是比较合适的初始值）

    for file_path in file_paths:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_names.append(file_name)

        abs_errors = load_data(file_path)
        all_abs_errors.append(abs_errors)

        if len(abs_errors) > 0:
            output_path = os.path.join(output_dir, f'abs_error_smoothed_{file_name}.png')
            create_single_error_chart(abs_errors, file_name, output_path, smooth_window=smooth_window)

    # 创建比较图表
    comparison_path = os.path.join(output_dir, 'comparison_smoothed_abs_errors.png')
    create_comparison_chart(all_abs_errors, file_names, comparison_path, smooth_window=smooth_window)

    print(f"所有平滑后的图表已生成并保存至: {output_dir}")


if __name__ == "__main__":
    main()