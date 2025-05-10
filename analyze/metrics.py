import numpy as np
from typing import List, Dict, Tuple, Any
from scipy.spatial.distance import euclidean
import json
from sentence_transformers import SentenceTransformer
import re
import torch
import logging
MODEL_PATH = "../../models/paraphrase-multilingual-MiniLM-L12-v2"
class OpinionControlEvaluator:
    """观点控制评估系统"""
    
    def __init__(self, llm_evaluator=None, embedding_model=None):
        """初始化评估器
        
        参数:
            llm_evaluator: 用于评估和提取观点的大模型
            embedding_model: 用于计算语义相似度的嵌入模型
        """
        # 如果未提供嵌入模型，则加载默认模型

        self.embedding_model = SentenceTransformer(MODEL_PATH)            
        self.llm = llm_evaluator
        
    def extract_paragraphs(self, article_text: str) -> List[str]:
        """从文章中提取段落
        
        参数:
            article_text: 文章文本
            
        返回:
            段落列表
        """
        # 简单的段落分割，基于空行或缩进
        paragraphs = re.split(r'\n\s*\n', article_text)
        # 过滤掉空段落
        return [p.strip() for p in paragraphs if p.strip()]
    
    def extract_opinions_with_llm(self, article_text: str) -> List[Dict]:
        """使用大模型从文章中提取观点及其参数
        
        参数:
            article_text: 文章文本
            
        返回:
            提取的观点列表，每个观点包含文本和参数估计
        """
        prompt = f"""
        请从以下文章中提取所有表达的观点，并为每个观点评估以下参数：
        - 偏见度(bias)：从-1(非常负面)到1(非常正面)，0表示中立
        - 情感强度(emotion)：从-1(非常消极)到1(非常积极)，0表示中性
        - 夸张程度(exaggeration)：从0(客观陈述)到1(高度夸张)
        
        文章内容：
        {article_text}
        
        请以JSON格式返回提取的观点，格式如下：
        [
            {{"text": "观点1内容", "bias": 0.5, "emotion": 0.7, "exaggeration": 0.3}},
            {{"text": "观点2内容", "bias": -0.8, "emotion": -0.6, "exaggeration": 0.4}}
        ]
        只返回JSON格式数据，不要添加任何额外文本、解释或标记。
        """
        
        # 使用大模型进行评估
        if self.llm is not None:
            try:
                print("开始使用大模型提取观点...")
                response = self.llm.generate(prompt)
                if not response:
                    print("大模型返回为空")
                    return []  # 直接返回空列表，不添加默认观点
                
                # 提取JSON部分并解析
                json_str = self._extract_json_from_text(response)
                
                if json_str:
                    opinions = json.loads(json_str)
                    print(f"成功提取 {len(opinions)} 个观点")
                    # 确保每个观点都有必要的字段
                    for op in opinions:
                        if "bias" not in op:
                            op["bias"] = 0.0
                        if "emotion" not in op:
                            op["emotion"] = 0.0
                        if "exaggeration" not in op:
                            op["exaggeration"] = 0.3
                        # 不再需要paragraph字段
                    return opinions
                else:
                    print(f"未能从响应中提取有效的JSON，原始响应：{response[:200]}...")
                    return []
            except Exception as e:
                print(f"提取观点时出错: {str(e)}")
                return []
        else:
            print("未提供大模型，无法提取观点")
            return []
    

    def _extract_json_from_text(self, text):
        """从文本中提取JSON字符串"""
        if not text:
            print("警告: 收到空响应")
            return None
        
        print(f"尝试从以下响应中提取JSON:\n{text[:200]}...")
        
        # 尝试多种提取方法
        extracted = None
        
        # 方法1: 尝试寻找JSON数组的起始和结束位置
        try:
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = text[start_idx:end_idx+1]
                # 验证是否是有效的JSON
                json.loads(json_str)
                extracted = json_str
                print("方法1成功提取JSON")
        except Exception as e:
            print(f"方法1提取JSON失败: {e}")
        
        # 如果方法1失败，尝试方法2
        if not extracted:
            try:
                import re
                json_pattern = r'\[\s*\{.*\}\s*\]'
                matches = re.search(json_pattern, text, re.DOTALL)
                if matches:
                    json_str = matches.group(0)
                    json.loads(json_str)
                    extracted = json_str
                    print("方法2成功提取JSON")
            except Exception as e:
                print(f"方法2提取JSON失败: {e}")
        
        # 如果方法2也失败，尝试方法3
        if not extracted:
            try:
                # 尝试手动构建JSON
                result = []
                in_object = False
                current_obj = ""
                
                lines = text.split('\n')
                for line in lines:
                    if '{' in line and not in_object:
                        in_object = True
                        current_obj = line[line.find('{'):]
                    elif in_object:
                        current_obj += line
                        if '}' in line:
                            in_object = False
                            try:
                                # 尝试解析单个对象
                                obj = json.loads(current_obj)
                                result.append(obj)
                                current_obj = ""
                            except:
                                pass
                
                if result:
                    extracted = json.dumps(result)
                    print("方法3成功提取JSON")
            except Exception as e:
                print(f"方法3提取JSON失败: {e}")
        
        return extracted
    def calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的语义相似度
        
        参数:
            text1, text2: 待比较的文本
            
        返回:
            相似度分数 [0,1]
        """
        emb1 = self.embedding_model.encode(text1)
        emb2 = self.embedding_model.encode(text2)
        
        # 计算余弦相似度
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def calculate_position_similarity(self, para_idx: int, opinion_para: int, total_paras: int) -> float:
        """计算位置相似度
        
        参数:
            para_idx: 段落索引
            opinion_para: 观点所在的段落索引
            total_paras: 总段落数
            
        返回:
            位置相似度分数 [0,1]
        """
        # 如果段落索引与观点段落索引完全匹配，返回1
        if para_idx == opinion_para:
            return 1.0
        
        # 否则，根据距离计算相似度
        distance = abs(para_idx - opinion_para)
        max_distance = total_paras - 1
        
        # 距离越近，相似度越高
        if max_distance == 0:  # 只有一个段落的情况
            return 1.0
        
        return 1.0 - (distance / max_distance)
    
    def calculate_key_phrase_similarity(self, paragraph: str, opinion_text: str) -> float:
        """计算关键词匹配度
        
        参数:
            paragraph: 段落文本
            opinion_text: 观点文本
            
        返回:
            关键词匹配分数 [0,1]
        """
        # 提取观点中的关键词（简单实现，可以用更复杂的NLP技术）
        opinion_words = set(re.findall(r'\w+', opinion_text.lower()))
        paragraph_words = set(re.findall(r'\w+', paragraph.lower()))
        
        if not opinion_words:
            return 0.0
        
        # 计算重叠比例
        overlap = len(opinion_words.intersection(paragraph_words))
        return overlap / len(opinion_words)
    
    def mapping_confidence(self, paragraph: str, opinion: Dict, para_idx: int, total_paras: int, 
                          alpha: float=0.6, beta: float=0.3, gamma: float=0.1) -> float:
        """计算段落与观点的映射置信度
        
        参数:
            paragraph: 段落文本
            opinion: 观点字典
            para_idx: 段落索引
            total_paras: 总段落数
            alpha, beta, gamma: 权重系数
            
        返回:
            映射置信度分数 [0,1]
        """
        S_embed = self.calculate_embedding_similarity(paragraph, opinion["text"])
        S_pos = self.calculate_position_similarity(para_idx, opinion.get("paragraph", 0), total_paras)
        S_key = self.calculate_key_phrase_similarity(paragraph, opinion["text"])
        
        return alpha * S_embed + beta * S_pos + gamma * S_key
    
        
    def calculate_pci(self, preset_params: Dict, realized_params: Dict) -> float:
        """计算参数一致性指标"""
        # 检查参数是否存在
        required_params = ["bias", "emotion", "exaggeration"]
        for param in required_params:
            if param not in preset_params or param not in realized_params:
                print(f"警告: 缺少必要参数 {param}")
                # 设置默认值
                preset_params[param] = preset_params.get(param, 0.0)
                realized_params[param] = realized_params.get(param, 0.0)
        
        # 构建参数向量
        P = np.array([preset_params["bias"], preset_params["emotion"], preset_params["exaggeration"]])
        R = np.array([realized_params["bias"], realized_params["emotion"], realized_params["exaggeration"]])
        
        # 计算欧几里得距离
        distance = np.linalg.norm(P - R)
        
        # 最大可能距离（对于[-1,1]范围的参数）
        max_distance = 2 * np.sqrt(3)
        
        # 计算PCI
        pci = 1 - (distance / max_distance)
        return pci
    
    def calculate_tcpi(self, preset_distribution: Dict, actual_distribution: Dict) -> float:
        """计算类型控制精确度指标
        
        参数:
            preset_distribution: 预设分布 {"positive": x, "neutral": y, "negative": z}
            actual_distribution: 实际分布 {"positive": x', "neutral": y', "negative": z'}
            
        返回:
            TCPI值 [0,1]
        """
        types = ["positive", "neutral", "negative"]
        
        # 计算绝对差异的总和
        diff_sum = sum(abs(preset_distribution.get(t, 0) - actual_distribution.get(t, 0)) for t in types)
        
        # 计算TCPI
        tcpi = 1 - (diff_sum / 2)  # 除以2是归一化，因为最大总差值为2
        return tcpi
    
    def calculate_pcpm(self, mappings: List[Tuple], paragraphs: List[str], 
                    preset_opinions: List[Dict], extracted_opinions: List[Dict]) -> Dict:
        """计算参数控制精确度矩阵"""
        # 按观点类型分组
        type_pci = {"positive": [], "neutral": [], "negative": []}
        
        print(f"计算PCPM: 映射数={len(mappings)}, 预设观点数={len(preset_opinions)}, 提取观点数={len(extracted_opinions)}")
        
        if not mappings:
            print("警告: 没有成功的观点映射，PCPM将为0")
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        
        for para_idx, op_idx, confidence in mappings:
            try:
                if op_idx >= len(preset_opinions):
                    print(f"错误: 观点索引{op_idx}超出预设观点数组范围")
                    continue
                    
                preset_op = preset_opinions[op_idx]
                
                # 对于提取的观点，使用段落索引找到对应观点
                # 这里需要确保段落索引在范围内
                if para_idx >= len(extracted_opinions):
                    print(f"错误: 段落索引{para_idx}超出提取观点数组范围")
                    continue
                    
                extracted_op = extracted_opinions[para_idx]
                
                # 确定观点类型
                bias = preset_op.get("bias", 0)
                if bias > 0.3:
                    op_type = "positive"
                elif bias < -0.3:
                    op_type = "negative"
                else:
                    op_type = "neutral"
                
                # 检查并打印参数
                print(f"预设观点{op_idx}: {preset_op}")
                print(f"提取观点{para_idx}: {extracted_op}")
                
                # 计算PCI并添加到对应类型
                pci = self.calculate_pci(preset_op, extracted_op)
                type_pci[op_type].append(pci)
                print(f"观点类型: {op_type}, PCI: {pci:.4f}")
            except Exception as e:
                print(f"计算单个观点PCI时出错: {str(e)}")
        
        # 计算每种类型的平均PCI
        pcpm = {}
        for op_type, pci_list in type_pci.items():
            if pci_list:
                pcpm[op_type] = sum(pci_list) / len(pci_list)
            else:
                pcpm[op_type] = 0.0
                print(f"警告: 没有{op_type}类型的观点映射")
                
        return pcpm
        
    def calculate_ooci(self, tcpi: float, pcpm: Dict, preset_distribution: Dict, 
                      lambda1: float=0.4, lambda2: float=0.6) -> float:
        """计算观点总体控制指标
        
        参数:
            tcpi: 类型控制精确度指标
            pcpm: 参数控制精确度矩阵
            preset_distribution: 预设分布 {"positive": x, "neutral": y, "negative": z}
            lambda1, lambda2: 权重系数
            
        返回:
            OOCI值 [0,1]
        """
        # 计算加权参数一致性分数
        weighted_pci = 0
        for op_type, weight in preset_distribution.items():
            weighted_pci += pcpm.get(op_type, 0) * weight
        
        # 计算OOCI
        ooci = lambda1 * tcpi + lambda2 * weighted_pci
        return ooci
    
    def calculate_coverage(self, mappings: List[Tuple], total_opinions: int) -> float:
        """计算观点覆盖率
        
        参数:
            mappings: 成功映射的关系列表
            total_opinions: 总观点数
            
        返回:
            覆盖率 [0,1]
        """
        # 计算成功映射的唯一观点数
        mapped_opinions = set(op_idx for _, op_idx, _ in mappings)
        
        # 计算覆盖率
        if total_opinions == 0:
            return 0.0
        
        return len(mapped_opinions) / total_opinions
    
    def _calculate_text_similarity(self, text1, text2):
        """计算两段文本的语义相似度
        
        参数:
            text1: 第一段文本
            text2: 第二段文本
            
        返回:
            相似度得分 (0-1)
        """
            # 使用已有的embedding_model计算余弦相似度
        emb1 = self.embedding_model.encode(text1)
        emb2 = self.embedding_model.encode(text2)
        
        # 计算余弦相似度
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cos_sim)

    def evaluate_opinion_control_text_based(self, article_text, preset_opinions, preset_ratios=None):
        """基于文本匹配的观点控制评估方法，专注于提取观点的实际指标值
        
        参数:
            article_text: 文章文本
            preset_opinions: 预设观点列表
            preset_ratios: 预设情感比例
            
        返回:
            评估指标字典
        """
        try:
            # 从文章中提取观点
            extracted_opinions = self.extract_opinions_with_llm(article_text)
            
            if not extracted_opinions:
                logging.warning("未能从文章中提取到有效观点")
                return {
                    "Opinion_Coverage": 0.0,
                    "PCI": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
                    "PCI_Overall": 0.0,
                    "Actual_Distribution": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
                    "Preset_Distribution": preset_ratios or {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
                }
            
            # 匹配提取的观点与预设观点
            matched_pairs = []  # [(extracted_opinion, preset_opinion), ...]
            matched_preset_ids = set()
            
            for ext_op in extracted_opinions:
                best_match = None
                best_score = 0.0
                
                for preset_op in preset_opinions:
                    # 如果这个预设观点已经匹配过了，跳过
                    preset_id = preset_op.get("id", "")
                    if preset_id in matched_preset_ids:
                        continue
                    
                    # 计算文本相似度 (简化版，实际应使用更复杂的相似度计算)
                    ext_text = ext_op.get("text", "").lower()
                    preset_text = preset_op.get("text", "").lower()
                    
                    # 简单的相似度计算 (可替换为更复杂的算法)
                    if ext_text and preset_text:
                        similarity = self._calculate_text_similarity(ext_text, preset_text)
                        
                        if similarity > 0.6 and similarity > best_score:  # 设置一个阈值
                            best_score = similarity
                            best_match = preset_op
                
                if best_match:
                    matched_pairs.append((ext_op, best_match))
                    matched_preset_ids.add(best_match.get("id", ""))
            
            # 计算覆盖率
            coverage = len(matched_pairs) / len(preset_opinions) if preset_opinions else 0.0
            
            # 分析提取观点的实际情感分布
            actual_sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            
            # 计算参数一致性指数(PCI)
            pci_by_type = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            type_counts = {"positive": 0, "neutral": 0, "negative": 0}
            
            for ext_op, preset_op in matched_pairs:
                # 获取提取观点的实际指标值 (这些应该是从提取观点中计算得到的)
                actual_bias = ext_op.get("bias", 0.0)
                actual_emotion = ext_op.get("emotion", 0.0)
                actual_exaggeration = ext_op.get("exaggeration", 0.0)
                
                # 获取预设观点的目标指标值
                preset_bias = preset_op.get("bias", 0.0)
                preset_emotion = preset_op.get("emotion", 0.0)
                preset_exaggeration = preset_op.get("exaggeration", 0.0)
                
                # 确定观点类型 (基于实际bias，因为我们关心的是提取观点的实际表现)
                if actual_bias > 0.3:
                    sentiment_type = "positive"
                elif actual_bias < -0.3:
                    sentiment_type = "negative"
                else:
                    sentiment_type = "neutral"
                
                # 更新实际情感分布计数
                actual_sentiment_counts[sentiment_type] += 1
                
                # 计算参数差异
                bias_diff = abs(actual_bias - preset_bias)
                emotion_diff = abs(actual_emotion - preset_emotion)
                exaggeration_diff = abs(actual_exaggeration - preset_exaggeration)
                
                # 计算参数一致性得分 (1 - 归一化差异)
                pci = 1.0 - ((bias_diff / 2.0 + emotion_diff / 2.0 + exaggeration_diff / 1.0) / 3.0)
                pci_by_type[sentiment_type] += pci
                type_counts[sentiment_type] += 1
            
            # 计算每种情感类型的平均PCI
            for sentiment_type in pci_by_type:
                if type_counts[sentiment_type] > 0:
                    pci_by_type[sentiment_type] /= type_counts[sentiment_type]
            
            # 计算总体PCI
            total_matched = sum(type_counts.values())
            pci_overall = sum(pci_by_type[t] * type_counts[t] for t in pci_by_type) / total_matched if total_matched > 0 else 0.0
            
            # 计算实际情感分布
            total_count = sum(actual_sentiment_counts.values())
            actual_distribution = {
                k: v / total_count if total_count > 0 else 0.0 
                for k, v in actual_sentiment_counts.items()
            }
            
            # 准备结果
            metrics = {
                "Opinion_Coverage": coverage,
                "PCI": pci_by_type,
                "PCI_Overall": pci_overall,
                "Actual_Distribution": actual_distribution,
                "Preset_Distribution": preset_ratios or {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
                "Extracted_Opinion_Count": len(extracted_opinions),
                "Matched_Opinion_Count": len(matched_pairs)
            }
            
            return metrics
        except Exception as e:
            logging.error(f"评估文章观点控制失败: {str(e)}")
            return {
                "Opinion_Coverage": 0.0,
                "PCI": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
                "PCI_Overall": 0.0,
                "Error": str(e)
            }
    
    def evaluate_global_control(self, article_text: str, 
                            expected_distributions=None, expected_params=None) -> Dict:
        """评估全文级控制的情况
        
        参数:
            article_text: 文章文本
            expected_distributions: 预期观点分布 {"positive": x, "neutral": y, "negative": z}
            expected_params: 各类型观点的预期参数 {"positive": {"bias": x, ...}, ...}
            
        返回:
            评估指标结果
        """
        # 1. 提取段落
        paragraphs = self.extract_paragraphs(article_text)
        
        # 2. 使用大模型提取实际观点
        extracted_opinions = self.extract_opinions_with_llm(article_text)
        
        # 3. 计算提取的观点类型分布
        extracted_type_counts = {"positive": 0, "neutral": 0, "negative": 0}
        for op in extracted_opinions:
            bias = op.get("bias", 0)
            if bias > 0.3:
                extracted_type_counts["positive"] += 1
            elif bias < -0.3:
                extracted_type_counts["negative"] += 1
            else:
                extracted_type_counts["neutral"] += 1
        
        total_extracted = sum(extracted_type_counts.values())
        actual_distribution = {k: v/total_extracted if total_extracted > 0 else 0 
                            for k, v in extracted_type_counts.items()}
        
        # 4. 预期分布，如果未提供则使用均匀分布
        if expected_distributions is None:
            preset_distribution = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
        else:
            preset_distribution = expected_distributions
        
        # 5. 计算TCPI - 类型控制精确度
        tcpi = self.calculate_tcpi(preset_distribution, actual_distribution)
        
        # 6. 设定预期参数，如果未提供则使用默认值
        if expected_params is None:
            global_expected_params = {
                "positive": {"bias": 0.7, "emotion": 0.6, "exaggeration": 0.3},
                "neutral": {"bias": 0.0, "emotion": 0.0, "exaggeration": 0.2},
                "negative": {"bias": -0.7, "emotion": -0.6, "exaggeration": 0.4}
            }
        else:
            global_expected_params = expected_params
        
        # 7. 计算实际参数与预期的一致性
        type_pcis = {"positive": [], "neutral": [], "negative": []}
        
        for op in extracted_opinions:
            bias = op.get("bias", 0)
            if bias > 0.3:
                op_type = "positive"
            elif bias < -0.3:
                op_type = "negative"
            else:
                op_type = "neutral"
            
            expected = global_expected_params[op_type]
            pci = self.calculate_pci(expected, op)
            type_pcis[op_type].append(pci)
        
        # 计算每种类型的平均PCI
        pcpm = {}
        for op_type, pci_list in type_pcis.items():
            if pci_list:
                pcpm[op_type] = sum(pci_list) / len(pci_list)
            else:
                pcpm[op_type] = 0.0
        
        # 8. 计算OOCI
        ooci = self.calculate_ooci(tcpi, pcpm, preset_distribution)
        
        # 返回结果，移除了Dominant_Type和Dominant_Achievement
        return {
            "PCI": {op_type: pci for op_type, pci in pcpm.items()},
            "TCPI": tcpi,
            "OOCI": ooci,
            "Preset_Distribution": preset_distribution,
            "Actual_Distribution": actual_distribution,
            "Total_Extracted_Opinions": len(extracted_opinions)
        }