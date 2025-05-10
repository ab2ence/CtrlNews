import sys
import os
import json
import time
import logging
import concurrent.futures
from datetime import datetime
from tqdm import tqdm
import random

# 项目相关导入
from tools.duckduckgo_searchtool import DuckDuckGoSearch
from tools.lm import DeepSeekR1, DeepSeekV3
from article_reviewer import ArticleReviewer
from simple_reviewer import SimpleArticleReviewer

# 导入引力场相关类
from GravitionalField import (
    NewsGravityField, 
    RoleCoordinator, 
    SearchManager, 
    AnswerGenerator, 
    process_questions_with_search
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局大模型
LLMHandler = DeepSeekV3()

# 重构后的文章生成器类
class ArticleNewsWriter:
    """基于引力场的新闻文章生成器"""
    
    def __init__(self, topic, llm):
        """初始化文章生成器
        
        参数:
            topic: 新闻主题
            llm: 大语言模型
        """
        self.topic = topic
        self.llm = llm
        self.opinions = {}  # 存储观点 {node_id: 观点文本}
        self.nodes_info = {}  # 存储节点信息 {node_id: 节点信息}
        logger.info(f"文章生成器初始化完成，主题: {topic}")
    def add_opinions_from_gravity_field_listversion(self, gravity_field):
        """从简化的引力场模型添加观点
        
        参数:
            gravity_field: SimpleListModel 实例
        """
        # 适配 SimpleListModel
        for node_id in gravity_field.peripheral_nodes:
            if node_id in gravity_field.opinions:
                opinion_text = gravity_field.opinions[node_id]
                # 获取对应的问题
                question_text = next((q for nid, q in gravity_field.questions if nid == node_id), "")
                
                # 获取角色
                role = gravity_field.roles.get(node_id, "专家")
                
                # 获取控制参数（如果有的话）
                bias = 0.0
                emotion = 0.0
                exaggeration = 0.3
                
                # 添加观点
                self.add_opinion(
                    text=opinion_text,
                    id=node_id,
                    source_type="peripheral",
                    bias=bias,
                    emotion=emotion,
                    exaggeration=exaggeration,
                    role=role
                )
    def add_opinions_from_gravity_field(self, gravity_field):
        """从引力场中批量添加观点
        
        参数:
            gravity_field: 引力场实例
        """
        count = 0
        # 修改：使用 peripheral_nodes 替代 get_terminal_nodes()
        terminal_nodes = gravity_field.peripheral_nodes
        
        for node_id, node in gravity_field.nodes.items():
            # 检查节点是否有观点
            if node.current_opinion and "text" in node.current_opinion:
                # 获取到该节点的路径信息（如果是末端节点）
                paths = []
                path_summary = ""
                if node_id in terminal_nodes:
                    # 修改：使用节点的 gravity_paths 属性替代 get_paths_to_terminal 方法
                    paths = node.gravity_paths
                    # 创建路径摘要（如果有路径）
                    if paths:
                        longest_path = max(paths, key=len)
                        path_questions = [gravity_field.nodes[path_id].question_text 
                                        for path_id in longest_path if path_id != node_id]
                        if path_questions:
                            path_summary = " → ".join([q[:30] + "..." for q in path_questions])
                
                # 创建节点信息字典
                node_info = {
                    "question": node.question_text,
                    "answer": node.answer_text,
                    "source_role": node.source_role,
                    "sentiment": node.current_opinion.get("actual_sentiment", 
                                node.current_opinion.get("intended_sentiment", 0)),
                    "mass": node.mass,
                    "relevance": node.properties.get('relevance', 0.0),
                    "is_terminal": node_id in terminal_nodes,
                    "path_aware": node.current_opinion.get("path_aware", False),
                    "path_summary": path_summary,
                    "paths": paths
                }
                
                # 添加观点
                self.opinions[node_id] = node.current_opinion["text"]
                self.nodes_info[node_id] = node_info
                count += 1
                
                # 记录末端节点的路径感知观点
                if node_id in terminal_nodes and node.current_opinion.get("path_aware", False):
                    logger.debug(f"添加路径感知外围节点观点: {node.question_text[:30]}...")
        
        logger.info(f"从引力场添加了 {count} 个观点，包含 {len([n for n in self.nodes_info.values() if n.get('is_terminal', False)])} 个外围节点观点")
        return count
    def get_ordered_opinions(self, max_opinions=None, prioritize_terminals=True):
        """获取按重要性排序的观点
        
        参数:
            max_opinions: 最大观点数，默认None表示全部
            prioritize_terminals: 是否优先考虑末端节点，默认为True
            
        返回:
            排序后的(node_id, opinion, node_info)列表
        """
        # 按节点质量排序
        sorted_opinions = []
        for node_id, opinion in self.opinions.items():
            if node_id in self.nodes_info:
                node_info = self.nodes_info[node_id]
                # 使用mass作为主要排序标准
                mass = node_info.get("mass", 0.5)
                relevance = node_info.get("relevance", 0.0)
                
                # 末端节点加权
                is_terminal = node_info.get("is_terminal", False)
                path_aware = node_info.get("path_aware", False)
                
                # 权重调整
                terminal_bonus = 1.5 if is_terminal else 1.0
                path_aware_bonus = 1.2 if path_aware else 1.0
                
                # 综合得分
                score = (0.7 * mass + 0.3 * relevance) * terminal_bonus * path_aware_bonus
                
                sorted_opinions.append((node_id, opinion, node_info, score))
        
        # 按得分降序排序
        sorted_opinions.sort(key=lambda x: x[3], reverse=True)
        
        # 如果指定了最大数量，进行截取
        if max_opinions and len(sorted_opinions) > max_opinions:
            # 如果优先考虑末端节点，确保有足够比例的末端节点
            if prioritize_terminals:
                terminal_opinions = [o for o in sorted_opinions if o[2].get('is_terminal', False)]
                non_terminal_opinions = [o for o in sorted_opinions if not o[2].get('is_terminal', False)]
                
                # 计算末端节点的目标数量（至少50%）
                target_terminal_count = max(int(max_opinions * 0.5), 1)
                target_terminal_count = min(target_terminal_count, len(terminal_opinions))
                
                # 选择得分最高的末端节点和非末端节点
                selected_terminals = terminal_opinions[:target_terminal_count]
                remaining_slots = max_opinions - target_terminal_count
                selected_non_terminals = non_terminal_opinions[:remaining_slots]
                
                # 合并并按原始得分重新排序
                combined = selected_terminals + selected_non_terminals
                combined.sort(key=lambda x: x[3], reverse=True)
                sorted_opinions = combined
            else:
                # 普通截取，仅按得分
                sorted_opinions = sorted_opinions[:max_opinions]
        
        # 返回结果不包含得分
        return [(node_id, opinion, node_info) for node_id, opinion, node_info, _ in sorted_opinions]
    
    def get_topic_clusters(self, ordered_opinions, num_clusters=3):
        """将观点分组为主题聚类
        
        参数:
            ordered_opinions: 排序后的观点列表
            num_clusters: 聚类数量
            
        返回:
            聚类列表，每个聚类包含(node_id, opinion, node_info)
        """
        if not ordered_opinions:
            return []
            
        # 如果观点数不足，直接返回
        if len(ordered_opinions) <= num_clusters:
            return [[opinion] for opinion in ordered_opinions]
        
        # 优先考虑末端节点作为聚类核心
        terminal_opinions = [op for op in ordered_opinions if op[2].get('is_terminal', False)]
        non_terminal_opinions = [op for op in ordered_opinions if not op[2].get('is_terminal', False)]
        
        # 创建聚类
        clusters = [[] for _ in range(num_clusters)]
        
        # 首先分配末端节点作为每个聚类的核心
        used_terminals = 0
        for i in range(min(num_clusters, len(terminal_opinions))):
            clusters[i].append(terminal_opinions[used_terminals])
            used_terminals += 1
        
        # 如果末端节点不足，使用非末端节点补充
        for i in range(used_terminals, min(num_clusters, len(ordered_opinions))):
            if non_terminal_opinions:
                clusters[i].append(non_terminal_opinions.pop(0))
        
        # 然后分配剩余的末端节点
        remaining_terminals = terminal_opinions[used_terminals:]
        for i, opinion in enumerate(remaining_terminals):
            cluster_idx = i % num_clusters
            clusters[cluster_idx].append(opinion)
        
        # 最后分配剩余的非末端节点
        for i, opinion in enumerate(non_terminal_opinions):
            cluster_idx = i % num_clusters
            clusters[cluster_idx].append(opinion)
        
        return clusters
    
    # def generate_article(self, params):
    #     """生成文章
        
    #     参数:
    #         params: 生成参数，包含length等
            
    #     返回:
    #         生成的文章内容
    #     """
    #     # 获取参数
    #     length = params.get("length", 800)
        
    #     # 获取排序后的观点
    #     max_opinions = 10  # 限制使用的观点数量，避免过长
    #     ordered_opinions = self.get_ordered_opinions(max_opinions, prioritize_terminals=True)
        
    #     if not ordered_opinions:
    #         return {
    #             "content": f"无法生成关于{self.topic}的文章，因为没有有效的观点。",
    #             "success": False
    #         }
        
    #     # 统计终端节点和路径感知观点
    #     terminal_count = len([op for op in ordered_opinions if op[2].get('is_terminal', False)])
    #     path_aware_count = len([op for op in ordered_opinions if op[2].get('path_aware', False)])
    #     logger.info(f"文章生成使用 {len(ordered_opinions)} 个观点，其中末端节点 {terminal_count} 个，路径感知观点 {path_aware_count} 个")
        
    #     # 获取观点聚类
    #     opinion_clusters = self.get_topic_clusters(ordered_opinions, num_clusters=3)
        
    #     # 构建提示词
    #     prompt = self._build_article_prompt(opinion_clusters, length)
        
    #     try:
    #         # 生成文章
    #         logger.info(f"生成文章内容，主题: {self.topic}，目标长度: {length}")
    #         article_content = self.llm.generate(prompt)
            
    #         return {
    #             "content": article_content,
    #             "success": True,
    #             "used_opinions": len(ordered_opinions),
    #             "used_opinion_ids": [op[0] for op in ordered_opinions],  # 保存实际使用的观点ID列表
    #             "total_opinions": len(self.opinions),
    #             "terminal_opinions": terminal_count,
    #             "path_aware_opinions": path_aware_count,
    #             "params": params
    #         }
    #     except Exception as e:
    #         logger.error(f"生成文章失败: {str(e)}")
    #         return {
    #             "content": f"生成失败: {str(e)}",
    #             "success": False
    #         }
    def generate_article(self, params):
        """生成文章
        
        参数:
            params: 生成参数，包含length等
            
        返回:
            生成的文章内容
        """
        # 获取参数
        length = params.get("length", 800)
        
        # 获取排序后的观点
        max_opinions = 10  # 限制使用的观点数量，避免过长
        ordered_opinions = self.get_ordered_opinions(max_opinions, prioritize_terminals=True)
        
        if not ordered_opinions:
            return {
                "content": f"无法生成关于{self.topic}的文章，因为没有有效的观点。",
                "success": False
            }
        
        # 统计终端节点和路径感知观点
        terminal_count = len([op for op in ordered_opinions if op[2].get('is_terminal', False)])
        path_aware_count = len([op for op in ordered_opinions if op[2].get('path_aware', False)])
        logger.info(f"文章生成使用 {len(ordered_opinions)} 个观点，其中末端节点 {terminal_count} 个，路径感知观点 {path_aware_count} 个")
        
        # 获取观点聚类
        opinion_clusters = self.get_topic_clusters(ordered_opinions, num_clusters=3)
        
        # 构建提示词
        prompt = self._build_article_prompt(opinion_clusters, length)
        
        try:
            # 生成文章
            logger.info(f"生成文章内容，主题: {self.topic}，目标长度: {length}")
            article_content = self.llm.generate(prompt)
            
            # 创建完整的观点内容列表
            used_opinions_details = []
            for node_id, opinion_text, node_info in ordered_opinions:
                # 提取观点的各项参数（如果有）
                bias = 0.0
                emotion = 0.0
                exaggeration = 0.3
                role = node_info.get("source_role", "专家")
                
                # 对于末端节点，尝试提取情感参数
                if node_info.get("is_terminal", False):
                    sentiment = node_info.get("sentiment", 0.0)
                    # 将sentiment转换为bias（简单情况下可以直接使用）
                    bias = sentiment
                
                # 构建观点对象
                opinion_obj = {
                    "id": node_id,
                    "text": opinion_text,
                    "question": node_info.get("question", ""),
                    "role": role,
                    "bias": bias,
                    "emotion": emotion,
                    "exaggeration": exaggeration,
                    "is_terminal": node_info.get("is_terminal", False),
                    "path_aware": node_info.get("path_aware", False)
                }
                
                used_opinions_details.append(opinion_obj)
            
            return {
                "content": article_content,
                "success": True,
                "used_opinions": len(ordered_opinions),
                "used_opinion_ids": [op[0] for op in ordered_opinions],  # 保存实际使用的观点ID列表
                "used_opinions_details": used_opinions_details,  # 新增：完整观点详情
                "total_opinions": len(self.opinions),
                "terminal_opinions": terminal_count,
                "path_aware_opinions": path_aware_count,
                "params": params
            }
        except Exception as e:
            logger.error(f"生成文章失败: {str(e)}")
            return {
                "content": f"生成失败: {str(e)}",
                "success": False
            }
    def _build_article_prompt(self, opinion_clusters, length):
        """Build enhanced article generation prompt"""
        themes_desc = ""
        
        for i, cluster in enumerate(opinion_clusters, 1):
            themes_desc += f"\n\nTheme {i}:"
            
            for node_id, opinion, node_info in cluster:
                # Limit question and opinion length
                question = node_info.get("question", "Unknown question")
                question = question[:60] + "..." if len(question) > 60 else question
                
                opinion_text = opinion[:150] + "..." if len(opinion) > 150 else opinion
                
                node_type = []
                if node_info.get("is_terminal", False):
                    node_type.append("Terminal Node")
                if node_info.get("path_aware", False):
                    node_type.append("Path-Aware")
                
                node_mark = f"[{', '.join(node_type)}]" if node_type else ""
                
                path_info = ""
                if node_info.get("path_summary"):
                    path_summary = node_info.get("path_summary")
                    path_summary = path_summary[:100] + "..." if len(path_summary) > 100 else path_summary
                    path_info = f"\n    Path: {path_summary}"
                
                themes_desc += f"\n  - Question{node_mark}: {question}{path_info}\n    Opinion: {opinion_text}"
        
        prompt = f"""
        You are an experienced journalist skilled in crafting thought-provoking, high-quality articles. Please create a comprehensive, professional news article on the topic "{self.topic}".

        [ARTICLE REQUIREMENTS]
        Word count: Strictly maintain {length} words (must be between {int(length * 0.95)} and {int(length * 1.05)} words)
        
        The article must naturally incorporate the following themes and viewpoints: {themes_desc}
        
        [QUALITY GUIDELINES]
        
        1. Relevance
        - Maintain consistent focus on the central topic throughout the article
        - Ensure each paragraph directly advances the reader's understanding of the main theme
        - Avoid unnecessary tangents
        
        2. Breadth
        - Cover all significant aspects of the topic
        - Present multiple perspectives and stakeholder viewpoints
        - Address both immediate implications and broader context
        
        3. Depth
        - Go beyond surface information to explore underlying mechanisms and relationships
        - Analyze causes, effects, and connections between key elements
        - Include specific details, examples, and evidence
        - Explore nuances and complexities of the subject matter
        
        4. Novelty
        - Provide unique perspectives or fresh insights
        - Avoid clichés and conventional framing
        - Make meaningful connections between ideas
        - Challenge conventional thinking where appropriate
        
        5. Coherence
        - Construct a clear narrative arc with logical progression
        - Create smooth transitions between paragraphs
        - Ensure each paragraph builds upon previous content
        - Maintain consistent tone and perspective
        
        6. Language Expression
        - Use precise, vivid language that engages readers
        - Vary sentence structure for rhythm and readability
        - Balance technical terminology with accessibility
        - Employ a professional journalistic tone
        
        7. Topic Development
        - Begin with a compelling introduction establishing the topic's significance
        - Progressively develop ideas with increasing depth
        - Connect sub-themes to create a cohesive exploration
        - Conclude with a synthesis of key insights
        
        8. Intellectual Value
        - Provide genuinely thought-provoking perspectives
        - Include analysis that encourages deeper consideration
        - Connect the topic to broader trends, patterns, or principles
        - Leave readers with enhanced understanding
        
        [STRUCTURAL GUIDELINES]
        - Craft a headline that accurately captures the article's essence
        - Write an engaging introduction that establishes the article's importance
        - Develop 5-7 substantive body paragraphs that progressively explore different aspects
        - Ensure each paragraph (3-5 sentences) focuses on a single main idea with supporting details
        - Conclude with a paragraph that synthesizes key insights and offers forward-looking perspective
        
        [IMPORTANT NOTES]
        - Integrate the provided viewpoints naturally without explicitly referencing them as "opinions" or "themes"
        - Present information as journalistic reporting, not as a collection of perspectives
        - Strictly adhere to the required word count
        """
        
        return prompt