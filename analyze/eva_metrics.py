import os
import json
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import re
import argparse
import sys
from metrics import OpinionControlEvaluator
from tools.lm import DeepSeekV3_Ali  # 假设这是您的语言模型模块

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpinionEvaluationManager:
    """观点控制评估管理器"""
    
    def __init__(self, llm=None):
        """初始化评估管理器
        
        参数:
            llm: 大语言模型，用于观点提取
        """
        # 初始化LLM（如果没有提供）
        self.llm = llm
        if self.llm is None:
            try:
                self.llm = DeepSeekV3_Ali()
                logger.info("初始化大语言模型成功")
            except Exception as e:
                logger.error(f"初始化大语言模型失败: {str(e)}")
                self.llm = None
        
        # 初始化评估器
        self.evaluator = OpinionControlEvaluator(llm_evaluator=self.llm)
    def _load_opinions_from_file(self, opinion_file, default_params=None):
        """从观点文件加载完整观点内容，而非仅ID
        
        参数:
            opinion_file: 观点JSON文件路径
            default_params: 默认参数字典
                
        返回:
            观点列表，每个观点包含完整信息
        """
        try:
            with open(opinion_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 首先尝试从used_opinions_details获取完整观点
            if 'used_opinions_details' in data and data['used_opinions_details']:
                logger.info(f"从'used_opinions_details'字段加载了 {len(data['used_opinions_details'])} 个完整观点")
                return data['used_opinions_details']
                
            # 如果没有完整观点但有文本和ID
            elif 'used_opinions' in data:
                logger.info(f"从'used_opinions'字段加载了 {len(data['used_opinions'])} 个观点")
                return data['used_opinions']
                
            # 如果只有ID列表，这种情况下无法进行文本匹配评估
            elif 'used_opinion_ids' in data:
                logger.warning(f"观点文件只包含ID，无法进行基于文本的评估: {opinion_file}")
                return []
            
            else:
                logger.warning(f"未找到有效观点数据，请检查文件格式: {opinion_file}")
                return []
            
        except Exception as e:
            logger.error(f"加载观点文件失败: {opinion_file}, 错误: {str(e)}")
            return []
    def evaluate_node_level_control(self, articles_dir, sentiment_ratios=None, default_params=None, output_dir=None):
        """评估观点级控制方法（自动加载观点文件）
        
        参数:
            articles_dir: 文章根目录，需包含/opinion子目录
            sentiment_ratios: 情感比例字典，如 {'positive': 0.3, 'neutral': 0.4, 'negative': 0.3}
            default_params: 默认参数字典，用于覆盖或补充节点参数
            output_dir: 评估结果输出目录，默认为articles_dir/evaluation
        
        返回:
            (详细结果DataFrame, 平均结果DataFrame)
        """
        # 设置预设情感比例
        if sentiment_ratios is None:
            sentiment_ratios = {'positive': 0.3, 'neutral': 0.4, 'negative': 0.3}
        
        # 设置默认参数
        if default_params is None:
            default_params = {
                'positive': {'bias': 0.7, 'emotion': 0.6, 'exaggeration': 0.3},
                'neutral': {'bias': 0.0, 'emotion': 0.0, 'exaggeration': 0.2},
                'negative': {'bias': -0.7, 'emotion': -0.6, 'exaggeration': 0.5}
            }
        
        # 设置输出目录
        output_dir = output_dir or os.path.join(articles_dir, "evaluation")
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找所有txt文件
        article_files = glob(os.path.join(articles_dir, "*.txt"))
        logger.info(f"找到{len(article_files)}个文章文件")
        
        if not article_files:
            logger.error("未找到任何文章文件!")
            return None, None
        
        # 存储评估结果
        detailed_results = []
        
        for article_path in tqdm(article_files, desc="评估观点级控制"):
            filename = os.path.basename(article_path)
            base_name = os.path.splitext(filename)[0]
            
            # 查找对应的观点文件
            opinion_file = os.path.join(articles_dir, "opinion", f"{base_name}_used_opinions.json")
            
            if not os.path.exists(opinion_file):
                logger.warning(f"没有找到对应的观点文件: {opinion_file}")
                continue
            
            try:
                # 读取文章内容
                with open(article_path, 'r', encoding='utf-8') as f:
                    article_text = f.read()
                
                # 加载完整观点内容，而非仅ID
                preset_opinions = self._load_opinions_from_file(opinion_file)
                
                if not preset_opinions:
                    logger.warning(f"观点文件中没有有效观点内容: {opinion_file}")
                    continue
            
                
                # 使用基于文本匹配的评估方法
                metrics = self.evaluator.evaluate_opinion_control_text_based(
                    article_text, 
                    preset_opinions,
                    sentiment_ratios  # 使用实际计算的观点分布
                )
                
                # 准备结果
                result = {
                    "文件名": filename,
                    "评估类型": "观点级控制",
                    "预设_正面比例": sentiment_ratios['positive'],
                    "预设_中性比例": sentiment_ratios['neutral'],
                    "预设_负面比例": sentiment_ratios['negative'],
                }
                
                # 添加评估指标
                if metrics:
                    # PCI信息
                    pci_data = metrics.get("PCI", {})
                    for opinion_type in ["positive", "neutral", "negative"]:
                        result[f"PCI_{opinion_type}"] = pci_data.get(opinion_type, 0.0)
                    
                    # 其他指标
                    for key, value in metrics.items():
                        if key != "PCI" and not isinstance(value, dict) and not isinstance(value, list):
                            result[key] = value
                        elif key in ["Preset_Distribution", "Actual_Distribution"]:
                            for type_key, type_value in value.items():
                                result[f"{key}_{type_key}"] = type_value
                
                detailed_results.append(result)
                
            except Exception as e:
                logger.error(f"评估文件失败: {article_path}, 错误: {str(e)}")
        
        # 转换为DataFrame
        df_detailed = pd.DataFrame(detailed_results)
        
        # 计算平均结果
        if not df_detailed.empty:
            metrics_columns = [col for col in df_detailed.columns 
                              if col not in ["文件名", "评估类型"] and not col.startswith("Preset_") and not col.startswith("Actual_")]
            
            avg_results = [{
                "评估类型": "观点级控制",
                "文章数": len(df_detailed),
                "预设_正面比例": sentiment_ratios['positive'],
                "预设_中性比例": sentiment_ratios['neutral'],
                "预设_负面比例": sentiment_ratios['negative']
            }]
            
            for col in metrics_columns:
                if col in df_detailed.columns:
                    avg_results[0][col] = df_detailed[col].mean()
            
            df_avg = pd.DataFrame(avg_results)
            
            # 保存结果
            detailed_file = os.path.join(output_dir, "node_level_control_detailed.csv")
            avg_file = os.path.join(output_dir, "node_level_control_average.csv")
            
            df_detailed.to_csv(detailed_file, index=False, encoding="utf-8")
            df_avg.to_csv(avg_file, index=False, encoding="utf-8")
            
            # 保存评估使用的参数
            params_info = {
                "sentiment_ratios": sentiment_ratios,
                "default_params": default_params
            }
            with open(os.path.join(output_dir, "node_level_params.json"), 'w', encoding='utf-8') as f:
                json.dump(params_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"观点级控制评估结果已保存至: {detailed_file} 和 {avg_file}")
            
            return df_detailed, df_avg
        else:
            logger.warning("没有生成任何有效的评估结果")
            return None, None
    
    def evaluate_global_control(self, articles_dir, sentiment_ratios=None, 
                            sentiment_params=None, output_dir=None):
        """评估全文级控制方法（使用大模型提取观点）
        
        参数:
            articles_dir: 文章根目录
            sentiment_ratios: 情感比例字典，如 {'positive': 0.3, 'neutral': 0.4, 'negative': 0.3}
            sentiment_params: 情感参数字典，如 {'positive': {'bias': 0.7, ...}, ...}
            output_dir: 评估结果输出目录，默认为articles_dir/metrics
        
        返回:
            (详细结果DataFrame, 平均结果DataFrame)
        """
        # 设置默认情感比例
        if sentiment_ratios is None:
            sentiment_ratios = {
                "positive": 0.33,
                "neutral": 0.34,
                "negative": 0.33
            }
        
        # 设置默认的情感参数
        if sentiment_params is None:
            sentiment_params = {
                "positive": {"bias": 0.7, "emotion": 0.6, "exaggeration": 0.3},
                "neutral": {"bias": 0.0, "emotion": 0.0, "exaggeration": 0.2},
                "negative": {"bias": -0.7, "emotion": -0.6, "exaggeration": 0.5}
            }
        
        # 设置输出目录
        output_dir = output_dir or os.path.join(articles_dir, "metrics")
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找所有txt文件
        article_files = glob(os.path.join(articles_dir, "*.txt"))
        logger.info(f"找到{len(article_files)}个文章文件")
        
        if not article_files:
            logger.error("未找到任何文章文件!")
            return None, None
        
        # 存储评估结果
        detailed_results = []
        
        # 处理每个文章文件
        for article_path in tqdm(article_files, desc="评估全文级控制"):
            filename = os.path.basename(article_path)
            
            try:
                # 读取文章内容
                with open(article_path, 'r', encoding='utf-8') as f:
                    article_text = f.read()
                
                # 使用evaluate_global_control，但传递预期参数
                metrics = self.evaluator.evaluate_global_control(
                    article_text, 
                    expected_distributions=sentiment_ratios,
                    expected_params=sentiment_params
                )
                
                # 准备结果 - 移除了预设_主导情感字段
                result = {
                    "文件名": filename,
                    "评估类型": "全文级控制",
                    "预设_正面比例": sentiment_ratios['positive'],
                    "预设_中性比例": sentiment_ratios['neutral'],
                    "预设_负面比例": sentiment_ratios['negative'],
                    "预设_正面偏见度": sentiment_params['positive']['bias'],
                    "预设_正面情感强度": sentiment_params['positive']['emotion'],
                    "预设_正面夸张程度": sentiment_params['positive']['exaggeration'],
                    "预设_中性偏见度": sentiment_params['neutral']['bias'],
                    "预设_中性情感强度": sentiment_params['neutral']['emotion'],
                    "预设_中性夸张程度": sentiment_params['neutral']['exaggeration'],
                    "预设_负面偏见度": sentiment_params['negative']['bias'],
                    "预设_负面情感强度": sentiment_params['negative']['emotion'],
                    "预设_负面夸张程度": sentiment_params['negative']['exaggeration']
                }
                
                # 添加评估指标
                if metrics:
                    # PCI信息
                    pci_data = metrics.get("PCI", {})
                    for opinion_type in ["positive", "neutral", "negative"]:
                        result[f"PCI_{opinion_type}"] = pci_data.get(opinion_type, 0.0)
                    
                    # 其他指标
                    for key, value in metrics.items():
                        if key != "PCI" and not isinstance(value, dict) and not isinstance(value, list):
                            result[key] = value
                        elif key in ["Preset_Distribution", "Actual_Distribution"]:
                            for type_key, type_value in value.items():
                                result[f"{key}_{type_key}"] = type_value
                
                detailed_results.append(result)
                
            except Exception as e:
                logger.error(f"评估文件失败: {article_path}, 错误: {str(e)}")
        
        # 转换为DataFrame
        df_detailed = pd.DataFrame(detailed_results)
        
        # 计算平均结果 - 移除了预设_主导情感
        if not df_detailed.empty:
            metrics_columns = [col for col in df_detailed.columns 
                            if col not in ["文件名", "评估类型"] and 
                            not col.startswith("预设_")]
            
            avg_results = [{
                "评估类型": "全文级控制",
                "文章数": len(df_detailed),
                "预设_正面比例": sentiment_ratios['positive'],
                "预设_中性比例": sentiment_ratios['neutral'],
                "预设_负面比例": sentiment_ratios['negative']
            }]
            
            for col in metrics_columns:
                if col in df_detailed.columns:
                    avg_results[0][col] = df_detailed[col].mean()
            
            df_avg = pd.DataFrame(avg_results)
            
            # 保存结果
            detailed_file = os.path.join(output_dir, "global_control_detailed.csv")
            avg_file = os.path.join(output_dir, "global_control_average.csv")
            
            df_detailed.to_csv(detailed_file, index=False, encoding="utf-8")
            df_avg.to_csv(avg_file, index=False, encoding="utf-8")
            
            # 保存评估使用的参数
            params_info = {
                "sentiment_ratios": sentiment_ratios,
                "sentiment_params": sentiment_params
            }
            with open(os.path.join(output_dir, "global_control_params.json"), 'w', encoding='utf-8') as f:
                json.dump(params_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"全文级控制评估结果已保存至: {detailed_file} 和 {avg_file}")
            
            return df_detailed, df_avg
        else:
            logger.warning("没有生成任何有效的评估结果")
            return None, None
        
    def _load_opinions_from_file(self, opinion_file, default_params=None):
        """从观点文件加载预设观点，并应用默认参数
        
        参数:
            opinion_file: 观点JSON文件路径
            default_params: 默认参数字典，用于补充节点参数
            
        返回:
            观点列表
        """
        try:
            with open(opinion_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取观点列表
            opinions = []
            
            # 根据您的文件格式提取观点
            if 'opinions' in data:
                opinions = data['opinions']
            elif 'used_opinions' in data:
                opinions = data['used_opinions']
            elif 'used_opinion_ids' in data:
                # 如果只有ID，我们需要构造观点对象
                opinion_ids = data.get('used_opinion_ids', [])
                
                # 为每个ID创建一个基本观点对象
                for i, op_id in enumerate(opinion_ids):
                    # 默认参数可以补充缺失的属性
                    opinion_type = self._determine_opinion_type(i, len(opinion_ids))
                    params = default_params.get(opinion_type, {}) if default_params else {}
                    
                    opinion = {
                        "id": op_id,
                        "text": f"Opinion {op_id}",  # 使用ID作为默认文本
                        "bias": params.get("bias", 0.0),
                        "emotion": params.get("emotion", 0.0),
                        "exaggeration": params.get("exaggeration", 0.3)
                    }
                    opinions.append(opinion)
            else:
                logger.warning(f"未找到观点数据，请检查文件格式: {opinion_file}")
                return []
            
            # 确保每个观点都包含必要的属性
            valid_opinions = []
            for op in opinions:
                if isinstance(op, dict):
                    # 如果有默认参数，为缺失的属性提供
                    if default_params:
                        # 根据当前bias判断类型
                        bias = op.get("bias", 0.0)
                        if bias > 0.3:
                            opinion_type = "positive"
                        elif bias < -0.3:
                            opinion_type = "negative"
                        else:
                            opinion_type = "neutral"
                        
                        type_params = default_params.get(opinion_type, {})
                        
                        # 补充缺失的属性
                        if "bias" not in op and "bias" in type_params:
                            op["bias"] = type_params["bias"]
                        if "emotion" not in op and "emotion" in type_params:
                            op["emotion"] = type_params["emotion"]
                        if "exaggeration" not in op and "exaggeration" in type_params:
                            op["exaggeration"] = type_params["exaggeration"]
                    
                    # 确保必要字段存在
                    if "text" not in op:
                        op["text"] = f"Opinion {len(valid_opinions)}"
                    
                    if all(key in op for key in ["bias", "emotion", "exaggeration"]):
                        valid_opinions.append(op)
                    else:
                        logger.warning(f"观点缺少必要属性，已忽略: {op}")
            
            logger.info(f"从文件加载了 {len(valid_opinions)} 个有效观点")
            return valid_opinions
            
        except Exception as e:
            logger.error(f"加载观点文件失败: {opinion_file}, 错误: {str(e)}")
            return []
    
    def _determine_opinion_type(self, index, total):
        """根据索引位置确定观点类型，用于创建默认观点
        
        参数:
            index: 观点在列表中的索引
            total: 观点总数
            
        返回:
            观点类型: "positive", "neutral", "negative"
        """
        # 默认分布：30% 正面，40% 中性，30% 负面
        if index < int(total * 0.3):
            return "positive"
        elif index < int(total * 0.7):
            return "neutral"
        else:
            return "negative"

def main():
    # 只保留必要的命令行参数
    parser = argparse.ArgumentParser(description="评估文章的观点控制效果")
    parser.add_argument("--mode", "-m", type=str, required=True, choices=["node", "global"],
                      help="评估模式: node=观点级控制, global=全文级控制")
    parser.add_argument("--articles_dir", "-d", type=str, required=True,
                      help="文章目录路径")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                      help="评估结果输出目录，默认为articles_dir/metrics")
    
    args = parser.parse_args()
    
    # 设置默认输出目录为metrics子文件夹
    if args.output_dir is None:
        args.output_dir = os.path.join(args.articles_dir, "metrics_global")
    
    # ======= 直接在代码中定义控制参数 =======
    
    # 情感比例相关参数
    positive_ratio = 0.33  # 正面观点比例
    neutral_ratio = 0.33   # 中性观点比例
    negative_ratio = 0.33  # 负面观点比例
    
    # 正面观点参数
    positive_bias = 0.8         # 正面观点的偏见度，范围[-1,1]
    positive_emotion = 0.7      # 正面观点的情感强度，范围[-1,1]
    positive_exaggeration = 0.3 # 正面观点的夸张程度，范围[0,1]
    
    # 中性观点参数
    neutral_bias = 0.0          # 中性观点的偏见度，范围[-1,1]
    neutral_emotion = 0.0       # 中性观点的情感强度，范围[-1,1]
    neutral_exaggeration = 0.1  # 中性观点的夸张程度，范围[0,1]
    
    # 负面观点参数
    negative_bias = -0.8        # 负面观点的偏见度，范围[-1,1]
    negative_emotion = -0.7     # 负面观点的情感强度，范围[-1,1]
    negative_exaggeration = 0.7 # 负面观点的夸张程度，范围[0,1]
    
    # 构造情感比例字典
    sentiment_ratios = {
        'positive': positive_ratio,
        'neutral': neutral_ratio,
        'negative': negative_ratio
    }
    
    # 归一化比例
    total = sum(sentiment_ratios.values())
    if abs(total - 1.0) > 0.01:  # 允许0.01的误差
        logger.warning(f"情感比例总和不为1（{total}），将自动归一化")
        sentiment_ratios = {k: v/total for k, v in sentiment_ratios.items()}
    
    # 构造情感参数字典
    sentiment_params = {
        'positive': {
            'bias': positive_bias,
            'emotion': positive_emotion,
            'exaggeration': positive_exaggeration
        },
        'neutral': {
            'bias': neutral_bias,
            'emotion': neutral_emotion,
            'exaggeration': neutral_exaggeration
        },
        'negative': {
            'bias': negative_bias,
            'emotion': negative_emotion,
            'exaggeration': negative_exaggeration
        }
    }
    
    # 初始化评估管理器
    evaluator = OpinionEvaluationManager()
    
    # 根据模式执行不同的评估
    if args.mode == "node":
        logger.info("执行观点级控制评估...")
        logger.info(f"使用情感比例: 正面={sentiment_ratios['positive']:.2f}, 中性={sentiment_ratios['neutral']:.2f}, 负面={sentiment_ratios['negative']:.2f}")
        evaluator.evaluate_node_level_control(
            articles_dir=args.articles_dir,
            sentiment_ratios=sentiment_ratios,
            default_params=sentiment_params,
            output_dir=args.output_dir
        )
    else:  # global模式
        logger.info(f"使用情感比例: 正面={sentiment_ratios['positive']:.2f}, 中性={sentiment_ratios['neutral']:.2f}, 负面={sentiment_ratios['negative']:.2f}")
        logger.info(f"使用控制参数: 正面({positive_bias:.1f},{positive_emotion:.1f},{positive_exaggeration:.1f}), "
                    f"中性({neutral_bias:.1f},{neutral_emotion:.1f},{neutral_exaggeration:.1f}), "
                    f"负面({negative_bias:.1f},{negative_emotion:.1f},{negative_exaggeration:.1f})")
        
        evaluator.evaluate_global_control(
            articles_dir=args.articles_dir,
            sentiment_ratios=sentiment_ratios,  # 使用和观点级相同的比例
            sentiment_params=sentiment_params,  # 使用和观点级相同的参数
            output_dir=args.output_dir
        )
    
    logger.info("评估完成!")

if __name__ == "__main__":
    main()