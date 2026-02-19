import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm


# 自动检测系统中可用的中文字体
def get_available_chinese_fonts():
    """获取系统中可用的中文字体列表"""
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
    # 如果没有找到中文字体，则使用默认字体，并尝试设置字体路径
    print("未找到可用的中文字体，图表中的中文可能无法正常显示。")
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def load_data(file_paths):
    """加载所有JSON文件中的数据"""
    all_data = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 处理数据，将1_mastery转换为浮点数，并提取acc
                processed_data = []
                for item in data:
                    try:
                        # 移除百分号并转换为浮点数
                        predicted = float(item.get('1_mastery', '0%').strip('%')) / 100.0
                        actual = float(item.get('acc', 0))
                        processed_data.append((predicted, actual))
                    except (ValueError, TypeError):
                        continue  # 跳过无法解析的数据点
                all_data.append(processed_data)
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            all_data.append([])  # 添加空列表保持索引一致
    return all_data


def calculate_errors(data):
    """计算一组数据的误差指标"""
    if not data:
        return None, None, None, None

    # 提取预测值和实际值
    predicted = np.array([item[0] for item in data])
    actual = np.array([item[1] for item in data])

    # 计算误差
    mae = np.mean(np.abs(predicted - actual))  # 平均绝对误差
    mse = np.mean((predicted - actual) ** 2)  # 均方误差
    rmse = np.sqrt(mse)  # 均方根误差

    # 计算误差分布（分位数）
    errors = predicted - actual
    error_quantiles = np.quantile(errors, [0.05, 0.25, 0.5, 0.75, 0.95])

    return mae, mse, rmse, error_quantiles


def create_distribution_chart(data, file_name, output_path=None):
    """为一组数据创建分布图，展示预测概率与实际概率的分布差异"""
    if not data:
        print(f"数据组 {file_name} 为空，无法创建图表")
        return None

    # 提取预测值和实际值
    predicted = np.array([item[0] for item in data])
    actual = np.array([item[1] for item in data])

    # 计算误差
    mae, mse, rmse, error_quantiles = calculate_errors(data)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{file_name} - 预测概率与实际概率分布对比', fontsize=16)

    # 1. 散点图
    ax1 = axes[0, 0]
    ax1.scatter(predicted, actual, alpha=0.7, color='#3498db', edgecolors='w', s=60)
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # 理想线 (y=x)
    ax1.set_title('预测概率 vs 实际概率')
    ax1.set_xlabel('预测概率')
    ax1.set_ylabel('实际概率')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    # 添加误差指标文本
    metrics_text = (f'MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\n\n'
                    f'误差分位数:\n'
                    f'  5%: {error_quantiles[0]:.4f}\n'
                    f' 25%: {error_quantiles[1]:.4f}\n'
                    f' 50%: {error_quantiles[2]:.4f}\n'
                    f' 75%: {error_quantiles[3]:.4f}\n'
                    f' 95%: {error_quantiles[4]:.4f}')
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 误差分布直方图
    ax2 = axes[0, 1]
    errors = predicted - actual
    ax2.hist(errors, bins=20, alpha=0.7, color='#e74c3c', edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)  # 理想线
    ax2.set_title('预测误差分布')
    ax2.set_xlabel('预测概率 - 实际概率')
    ax2.set_ylabel('频次')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 3. 预测概率分布
    ax3 = axes[1, 0]
    ax3.hist(predicted, bins=20, alpha=0.7, color='#2ecc71', edgecolor='black', label='预测概率')
    ax3.hist(actual, bins=20, alpha=0.7, color='#f39c12', edgecolor='black', label='实际概率')
    ax3.set_title('预测概率与实际概率分布')
    ax3.set_xlabel('概率值')
    ax3.set_ylabel('频次')
    ax3.set_xlim(0, 1)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax3.legend()

    # 4. 概率区间误差
    ax4 = axes[1, 1]
    # 将概率分为10个区间
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mean_errors = []
    std_errors = []

    for i in range(len(bins) - 1):
        mask = (predicted >= bins[i]) & (predicted < bins[i + 1])
        if np.sum(mask) > 0:
            bin_errors = predicted[mask] - actual[mask]
            mean_errors.append(np.mean(bin_errors))
            std_errors.append(np.std(bin_errors))
        else:
            mean_errors.append(0)
            std_errors.append(0)

    ax4.bar(bin_centers, mean_errors, width=0.08, color='#9b59b6', alpha=0.7,
            yerr=std_errors, capsize=5, edgecolor='black')
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)  # 理想线
    ax4.set_title('各概率区间的平均预测误差')
    ax4.set_xlabel('预测概率区间')
    ax4.set_ylabel('平均误差 (预测 - 实际)')
    ax4.set_xlim(-0.05, 1.05)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def main():
    """主函数"""
    # 直接在代码中填写十个文件的名称
    file_paths = [
        "memory/exam_after_memory1.json",
        "memory/exam_after_memory1_AS.json",
        "memory/exam_after_memory1_AS2.json"
    ]

    # 加载数据
    all_data = load_data(file_paths)

    # 为每组数据创建单独的图表
    for i, (data, file_path) in enumerate(zip(all_data, file_paths)):
        file_name = file_path.split('/')[-1].split('.')[0]  # 提取文件名
        print(f"正在生成 {file_name} 的图表...")
        fig = create_distribution_chart(data, file_name, f'distribution_{file_name}.png')
        if fig:
            plt.close(fig)  # 关闭图表以节省内存，除非你想一次性显示所有图表

    print("所有图表生成完成！")


if __name__ == "__main__":
    main()