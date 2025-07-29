import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import chi2
from typing import List, Dict, Tuple, Optional


class ProbabilityMetricsEvaluator:
    """增强版概率评估指标计算类，专为连续概率数据优化"""

    def __init__(self, total_questions: int = 10, use_rounding: bool = True,
                 num_bins: int = 10, hl_bins: int = 10, threshold: float = 0.5):
        """初始化评估器"""
        self.total_questions = total_questions
        self.use_rounding = use_rounding
        self.num_bins = num_bins
        self.hl_bins = hl_bins
        self.threshold = threshold  # 用于将连续概率转换为二分类标签的阈值

    def calculate_variance(self, mastery_str: str, acc: float) -> float:
        """计算预测概率与实际准确率之间的方差"""
        try:
            pred_prob = float(mastery_str.strip('%')) / 100.0
            return (pred_prob - acc) ** 2
        except ValueError:
            print(f"警告: 无法解析预测概率 '{mastery_str}'，返回无穷大")
            return float('inf')

    def calculate_brier_score(self, mastery_str: str, acc: float) -> float:
        """计算Brier分数"""
        if self.total_questions <= 0:
            raise ValueError("总题数必须为正整数")

        try:
            pred_prob = float(mastery_str.strip('%')) / 100.0
            correct = int(round(acc * self.total_questions)) if self.use_rounding else int(acc * self.total_questions)
            correct = max(0, min(correct, self.total_questions))
            incorrect = self.total_questions - correct
            return (correct * (pred_prob - 1) ** 2 + incorrect * pred_prob ** 2) / self.total_questions
        except (ValueError, TypeError) as e:
            print(f"警告: 计算Brier分数时出错: {e}，返回无穷大")
            return float('inf')

    def calculate_ece(self, pred_probs: np.ndarray, accuracies: np.ndarray,
                      adaptive: bool = False) -> float:
        """计算预期校准误差(ECE)"""
        if len(pred_probs) != len(accuracies):
            raise ValueError("预测概率数组和实际准确率数组长度不一致")

        if adaptive:
            sorted_indices = np.argsort(pred_probs)
            pred_sorted = pred_probs[sorted_indices]
            acc_sorted = accuracies[sorted_indices]
            bin_indices = np.array_split(np.arange(len(pred_sorted)), self.num_bins)
            ece = 0.0
            for bins in bin_indices:
                if len(bins) == 0:
                    continue
                conf = np.mean(pred_sorted[bins])
                acc = np.mean(acc_sorted[bins])
                frac = len(bins) / len(pred_sorted)
                ece += frac * abs(conf - acc)
            return ece
        else:
            bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
            ece = 0.0
            for i in range(self.num_bins):
                mask = (pred_probs >= bin_boundaries[i]) & (pred_probs < bin_boundaries[i + 1])
                if np.sum(mask) > 0:
                    conf = np.mean(pred_probs[mask])
                    acc = np.mean(accuracies[mask])
                    frac = np.sum(mask) / len(pred_probs)
                    ece += frac * abs(conf - acc)
            return ece

    def calculate_auc(self, pred_probs: np.ndarray, accuracies: np.ndarray) -> float:
        """计算AUC值，将连续概率转换为二分类标签"""
        # 将连续概率转换为二分类标签
        binary_labels = (accuracies >= self.threshold).astype(int)

        # 检查标签是否包含正负两类
        unique_labels = np.unique(binary_labels)
        if len(unique_labels) < 2:
            print(f"警告: AUC计算失败 - 二分类标签只有一种值: {unique_labels}")
            print(f"提示: 尝试调整阈值(当前阈值={self.threshold})或检查数据分布")
            return float('nan')

        try:
            return roc_auc_score(binary_labels, pred_probs)
        except Exception as e:
            print(f"警告: 计算AUC时出错: {e}")
            return float('nan')

    def calculate_auprc(self, pred_probs: np.ndarray, accuracies: np.ndarray) -> float:
        """计算AUPRC值，将连续概率转换为二分类标签"""
        # 将连续概率转换为二分类标签
        binary_labels = (accuracies >= self.threshold).astype(int)

        # 检查标签是否包含正负两类
        unique_labels = np.unique(binary_labels)
        if len(unique_labels) < 2:
            print(f"警告: AUPRC计算失败 - 二分类标签只有一种值: {unique_labels}")
            print(f"提示: 尝试调整阈值(当前阈值={self.threshold})或检查数据分布")
            return float('nan')

        try:
            precision, recall, _ = precision_recall_curve(binary_labels, pred_probs)
            return auc(recall, precision)
        except Exception as e:
            print(f"警告: 计算AUPRC时出错: {e}")
            return float('nan')

    def calculate_log_loss(self, pred_probs: np.ndarray, accuracies: np.ndarray) -> float:
        """计算Log Loss"""
        # 避免对数计算中的数值稳定性问题
        epsilon = 1e-15
        pred_probs = np.clip(pred_probs, epsilon, 1 - epsilon)

        try:
            return -np.mean(accuracies * np.log(pred_probs) + (1 - accuracies) * np.log(1 - pred_probs))
        except Exception as e:
            print(f"警告: 计算Log Loss时出错: {e}")
            return float('nan')

    def calculate_hosmer_lemeshow(self, pred_probs: np.ndarray, accuracies: np.ndarray) -> Dict:
        """计算Hosmer-Lemeshow测试统计量和p值，优化连续概率数据的分箱"""
        if self.hl_bins <= 0:
            raise ValueError("Hosmer-Lemeshow测试的分箱数必须为正整数")

        try:
            # 对连续概率数据使用自适应分箱
            sorted_indices = np.argsort(pred_probs)
            pred_sorted = pred_probs[sorted_indices]
            acc_sorted = accuracies[sorted_indices]
            bin_indices = np.array_split(np.arange(len(pred_sorted)), self.hl_bins)

            hl_statistic = 0.0
            valid_bins = 0

            for bins in bin_indices:
                if len(bins) == 0:
                    continue

                # 计算分箱内的观测平均概率和预期平均概率
                obs_mean = np.mean(acc_sorted[bins])
                exp_mean = np.mean(pred_sorted[bins])

                # 计算卡方统计量
                if exp_mean > 0 and exp_mean < 1:
                    # 计算观测值和期望值
                    n = len(bins)
                    obs_pos = obs_mean * n
                    exp_pos = exp_mean * n
                    obs_neg = n - obs_pos
                    exp_neg = n - exp_pos

                    # 累加卡方统计量
                    if exp_pos > 0:
                        hl_statistic += (obs_pos - exp_pos) ** 2 / exp_pos
                    if exp_neg > 0:
                        hl_statistic += (obs_neg - exp_neg) ** 2 / exp_neg
                    valid_bins += 1

            # 计算p值
            degrees_of_freedom = max(1, valid_bins - 2)

            # 防止统计量过大导致p值下溢
            if hl_statistic > 1e10:
                print(f"警告: HL统计量过大 ({hl_statistic:.2e})，预测概率可能过于集中")
                return {
                    'statistic': hl_statistic,
                    'p_value': 0.0,
                    'degrees_of_freedom': degrees_of_freedom,
                    'warning': '统计量过大，预测概率可能过于集中'
                }

            p_value = 1 - chi2.cdf(hl_statistic, degrees_of_freedom)

            return {
                'statistic': hl_statistic,
                'p_value': p_value,
                'degrees_of_freedom': degrees_of_freedom
            }
        except Exception as e:
            print(f"警告: 计算Hosmer-Lemeshow测试时出错: {e}")
            return {
                'statistic': float('nan'),
                'p_value': float('nan'),
                'degrees_of_freedom': self.hl_bins - 2,
                'error': str(e)
            }

    def load_data(self, file_path: str) -> List[Dict]:
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

    def parse_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """解析数据，提取预测概率和实际准确率"""
        pred_probs = []
        accuracies = []
        for item in data:
            if '1_mastery' in item:
                mastery = item['1_mastery']
                try:
                    p = float(mastery.strip('%')) / 100.0
                    pred_probs.append(p)
                except ValueError:
                    print(f"警告: 无法解析1_mastery值 '{mastery}'，跳过此样本")
                    continue
            else:
                print("警告: 样本缺少'1_mastery'字段，跳过此样本")
                continue

            if 'acc' in item:
                a = item['acc']
                if isinstance(a, (int, float)) and 0.0 <= a <= 1.0:
                    accuracies.append(a)
                else:
                    print(f"警告: acc值 '{a}' 不是有效数值或超出范围 [0, 1]，跳过此样本")
                    continue
            else:
                print("警告: 样本缺少'acc'字段，跳过此样本")
                continue

        return np.array(pred_probs), np.array(accuracies)

    def evaluate_file(self, file_path: str, adaptive_ece: bool = False) -> Dict:
        """评估单个文件的所有指标"""
        print(f"\n正在处理文件: {file_path}")
        data = self.load_data(file_path)
        if not data:
            print("没有数据可处理")
            return {}

        pred_probs, accuracies = self.parse_data(data)
        if len(pred_probs) == 0:
            print("解析后没有有效数据")
            return {}

        print(f"成功解析 {len(pred_probs)} 个有效样本")

        # 打印数据分布信息
        print(f"  预测概率范围: [{np.min(pred_probs):.4f}, {np.max(pred_probs):.4f}]")
        print(f"  实际概率范围: [{np.min(accuracies):.4f}, {np.max(accuracies):.4f}]")
        print(f"  预测概率分布: 均值={np.mean(pred_probs):.4f}, 标准差={np.std(pred_probs):.4f}")
        print(f"  实际概率分布: 均值={np.mean(accuracies):.4f}, 标准差={np.std(accuracies):.4f}")

        # 计算各项指标
        variances = [self.calculate_variance(f"{p * 100:.2f}%", a) for p, a in zip(pred_probs, accuracies)]
        avg_variance = np.mean(variances)

        brier_scores = [self.calculate_brier_score(f"{p * 100:.2f}%", a) for p, a in zip(pred_probs, accuracies)]
        avg_brier = np.mean(brier_scores)

        ece = self.calculate_ece(pred_probs, accuracies, adaptive_ece)
        mce = self.calculate_ece(pred_probs, accuracies, True)

        auc_score = self.calculate_auc(pred_probs, accuracies)
        auprc_score = self.calculate_auprc(pred_probs, accuracies)
        log_loss = self.calculate_log_loss(pred_probs, accuracies)

        hl_test = self.calculate_hosmer_lemeshow(pred_probs, accuracies)

        # 计算分类准确率（以0.5为阈值）
        binary_pred = (pred_probs >= 0.5).astype(int)
        binary_labels = (accuracies >= 0.5).astype(int)
        classification_acc = np.mean(binary_pred == binary_labels)

        return {
            'num_samples': len(pred_probs),
            'average_variance': avg_variance,
            'average_brier_score': avg_brier,
            'ece': ece,
            'mce': mce,
            'auc': auc_score,
            'auprc': auprc_score,
            'log_loss': log_loss,
            'hosmer_lemeshow': hl_test,
            'classification_accuracy': classification_acc,
            'pred_prob_mean': np.mean(pred_probs),
            'pred_prob_std': np.std(pred_probs),
            'actual_prob_mean': np.mean(accuracies),
            'actual_prob_std': np.std(accuracies),
            'threshold': self.threshold
        }

    def evaluate_multiple_files(self, file_paths: List[str], adaptive_ece: bool = True) -> Dict:
        """评估多个文件并汇总结果"""
        all_results = {}
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"错误：文件路径 '{file_path}' 不存在，跳过此文件")
                continue

            results = self.evaluate_file(file_path, adaptive_ece)
            if results:
                all_results[file_path] = results

        return all_results

    def print_summary(self, results: Dict) -> None:
        """打印评估结果汇总"""
        if not results:
            print("没有计算任何评估指标")
            return

        print("\n===== 汇总结果 =====")
        first_result = next(iter(results.values()))
        print(f"评估参数: 总题数={self.total_questions}, ECE分箱数={self.num_bins}, "
              f"Hosmer-Lemeshow分箱数={self.hl_bins}, 二分类阈值={first_result['threshold']}")

        # 打印表头
        """print("\n{:<32} {:<8} {:<8} {:<11} {:<10} {:<9} {:<9} {:<9} {:<12} {:<8} {:<10}".format(
            "文件名", "样本数量", "平均方差", "Brier分数", "ECE", "MCE", "AUC", "AUPRC", "Log Loss", "HL p值",
            "分类准确率"))"""
        print("\n{:<32} {:<8} {:<8} {:<11} {:<10} {:<9} {:<12} {:<10}".format(
            "文件名", "样本数量", "平均方差", "Brier分数", "ECE", "AUC", "Log Loss", "分类准确率"))
        print("-" * 180)

        # 打印每个文件结果
        for file_path, file_results in results.items():


            # 转换方差：(100 - 方差*100)，保留两位小数
            variance_converted = 100 - file_results['average_variance'] * 100
            # 转换Brier分数：(100 - Brier*100)，保留两位小数
            brier_converted = 100 - file_results['average_brier_score'] * 100



            ece_converted = 100 - file_results['ece'] * 100

            logloss_converted = 100 - file_results['log_loss'] * 10

            # 转换AUC和AUPRC为百分比，保留两位小数
            auc_percent = file_results['auc'] * 100 if not np.isnan(file_results['auc']) else float('nan')
            auprc_percent = file_results['auprc'] * 100 if not np.isnan(file_results['auprc']) else float('nan')

            # 格式化显示：nan时显示'-'，否则显示百分比
            auc_str = f"{auc_percent:.2f}" if not np.isnan(auc_percent) else '-'
            auprc_str = f"{auprc_percent:.2f}" if not np.isnan(auprc_percent) else '-'

            print(
                #"{:<38} {:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.4f} {:<10} {:<10} {:<10.2f} {:<10.2f} {:<12.2%}".format(
                "{:<38} {:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10} {:<10.2f} {:<12.2%}".format(
                    os.path.basename(file_path),
                    file_results['num_samples'],
                    variance_converted,
                    brier_converted,
                    ece_converted,
                    #file_results['mce'],
                    auc_str,
                    #auprc_str,
                    logloss_converted,
                    #file_results['hosmer_lemeshow']['p_value'],
                    file_results['classification_accuracy']
                ))

        # 计算总体统计
        valid_results = [r for r in results.values() if not np.isnan(r['auc'])]
        if valid_results:
            all_num = [r['num_samples'] for r in valid_results]
            all_var = [r['average_variance'] for r in valid_results]
            all_brier = [r['average_brier_score'] for r in valid_results]
            all_ece = [r['ece'] for r in valid_results]
            all_mce = [r['mce'] for r in valid_results]
            all_auc = [r['auc'] for r in valid_results]
            all_auprc = [r['auprc'] for r in valid_results]
            all_log_loss = [r['log_loss'] for r in valid_results]

            total = sum(all_num)
            weighted_var = sum(v * n / total for v, n in zip(all_var, all_num))
            weighted_brier = sum(b * n / total for b, n in zip(all_brier, all_num))
            weighted_ece = sum(e * n / total for e, n in zip(all_ece, all_num))
            weighted_mce = sum(m * n / total for m, n in zip(all_mce, all_num))
            weighted_auc = sum(a * n / total for a, n in zip(all_auc, all_num))
            weighted_auprc = sum(a * n / total for a, n in zip(all_auprc, all_num))
            weighted_log_loss = sum(l * n / total for l, n in zip(all_log_loss, all_num))

            print(
                "\n\n{:<35} {:<11} {:<10.2f} {:<10.2f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10} {:<12}".format(
                    "总体统计",
                    total,
                    weighted_var,
                    weighted_brier,
                    weighted_ece,
                    weighted_mce,
                    weighted_auc,
                    weighted_auprc,
                    weighted_log_loss,
                    "N/A",
                    "N/A"
                ))


def main():
    """主函数：配置并执行评估"""
    # 初始化评估器
    evaluator = ProbabilityMetricsEvaluator(
        total_questions=10,
        num_bins=10,
        hl_bins=10,
        threshold=0.7  # 可调整此参数，例如0.6或0.7，根据业务需求确定
    )

    # 定义要处理的文件列表
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
    ]

    # 执行评估
    results = evaluator.evaluate_multiple_files(file_paths, adaptive_ece=True)

    # 打印汇总结果
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()