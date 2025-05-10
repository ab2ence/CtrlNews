import hashlib
import json
import math
import time
import logging
import numpy as np
import sys
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import gc
from tools.lm import DeepSeekV3
from tools.duckduckgo_searchtool import DuckDuckGoSearch
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchManager:
    """Search manager for handling web search operations"""
    
    def __init__(self, max_results=5):
        """Initialize search manager
        
        Args:
            max_results: Maximum number of search results per query
        """
        self.max_results = max_results
        # Initialize DuckDuckGo search
        self.searcher = DuckDuckGoSearch(self.max_results)
    
    def search(self, query):
        """Perform search with the given query
        
        Args:
            query: Search query text
            
        Returns:
            List of search results
        """
        logger.info(f"Searching for: {query}")
        
        try:
            results = self.searcher(query)
            logger.info(f"Search returned {len(results)} results")
            return self.process_search_results(results)
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def process_search_results(self, results):
        """Process and format search results
        
        Args:
            results: Raw search results
            
        Returns:
            Processed results
        """
        processed_results = []
        
        for result in results:
            # 提取关键信息
            title = result.get("title", "").strip()
            url = result.get("url", "")
            
            # 处理不同格式的搜索结果
            # 1. 检查'snippet'键
            snippet = result.get("snippet", "").strip()
            
            # 2. 如果没有'snippet'键，检查'snippets'键（DuckDuckGoSearch返回的格式）
            if not snippet and "snippets" in result:
                snippets = result.get("snippets", [])
                if snippets and isinstance(snippets, list):
                    # 将所有片段合并为一个文本
                    snippet = " ".join(snippets)
            
            # 3. 如果仍然没有有效片段，尝试使用'description'键
            if not snippet:
                snippet = result.get("description", "").strip()
            
            # 过滤太短的结果
            if not snippet or len(snippet) < 10:
                logger.debug(f"过滤结果 '{title}': 片段太短或为空")
                continue
                
            # 标准化格式
            processed_result = {
                "title": title,
                "url": url,
                "snippet": snippet,
                "timestamp": time.time()
            }
            
            processed_results.append(processed_result)
        
        # 如果所有结果都被过滤，但有原始结果，选择最好的一个
        if not processed_results and results:
            logger.warning("所有搜索结果都被过滤，使用最佳可用结果")
            best_result = results[0]  # 默认选择第一个
            
            # 尝试找出有最多信息的结果
            for result in results:
                if (len(result.get("description", "")) > len(best_result.get("description", "")) or
                    (isinstance(result.get("snippets", []), list) and 
                    len(result.get("snippets", [])) > len(best_result.get("snippets", [])))):
                    best_result = result
            
            # 构建一个结果
            snippet = best_result.get("description", "")
            if not snippet and "snippets" in best_result:
                snippets = best_result.get("snippets", [])
                if snippets and isinstance(snippets, list):
                    snippet = " ".join(snippets)
            
            if not snippet:
                snippet = "No detailed content available"
            
            processed_results.append({
                "title": best_result.get("title", "No title").strip(),
                "url": best_result.get("url", ""),
                "snippet": snippet,
                "timestamp": time.time()
            })
            
        return processed_results


class AnswerGenerator:
    """Answer generator using search results to answer questions"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def generate_answer(self, question, search_results, detailed=False):
        """Generate answer based on search results
        
        Args:
            question: Question text
            search_results: List of search results
            detailed: Whether to generate detailed answer
            
        Returns:
            Generated answer
        """
        # 检查搜索结果是否为空
        if not search_results:
            # 搜索结果为空，使用大模型直接回答
            return self.generate_direct_answer(question, detailed)
            
        # 提取搜索结果文本
        snippets = []
        sources = []
        
        for result in search_results:
            if "snippet" in result and result["snippet"]:
                snippets.append(result["snippet"])
            if "url" in result and result["url"]:
                sources.append(result["url"])
                
        if not snippets:
            # 如果提取的片段为空，也使用大模型直接回答
            return self.generate_direct_answer(question, detailed)
            
        # 原有的答案生成代码保持不变...
        combined_text = "\n\n".join([f"Information {i+1}: {text}" for i, text in enumerate(snippets)])
        
        # Limit text length
        max_length = 6000 if detailed else 3000
        combined_text = combined_text[:max_length]
        max_words = 300 if detailed else 150
        
        if detailed:
            prompt = f"""
            Based on the following retrieved information, please provide a detailed answer to the question. Ensure the answer is comprehensive, accurate, and objective.

            Question: {question}

            Retrieved information:
            {combined_text}

            IMPORTANT: Your answer must be between {max_words-50} and {max_words} words long. Focus on the most relevant information and be concise.
            
            Please provide a thorough but concise answer, using bullet points where appropriate.
            """
        else:
            prompt = f"""
            Based on the following retrieved information, please answer the question concisely.

            Question: {question}

            Retrieved information:
            {combined_text}

            IMPORTANT: Your answer must be strictly under {max_words} words. Be direct and focus only on key information.
            """
            
        try:
            # 生成答案
            answer = self.llm.generate(prompt).strip()
            
            # 添加来源信息
            if sources and detailed:
                sources_text = "\n".join([f"- {s}" for s in sources[:3]])
                answer += f"\n\nReferences:\n{sources_text}"
                
            return answer
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            # 如果答案生成失败，尝试直接回答
            return self.generate_direct_answer(question, detailed)

    def generate_direct_answer(self, question, detailed=False):
        """当搜索结果为空时，使用大模型直接回答问题
        
        Args:
            question: 问题文本
            detailed: 是否生成详细答案
            
        Returns:
            生成的答案
        """
        max_words = 300 if detailed else 150
        
        if detailed:
            prompt = f"""
            Please answer this question based on your own knowledge. Provide a comprehensive, accurate, and objective response.
            
            Question: {question}
            
            IMPORTANT: 
            1. Your answer must be between {max_words-50} and {max_words} words long
            2. Be thorough but concise
            3. Use bullet points where appropriate
            4. If you're uncertain about specific facts, acknowledge the limitations of your knowledge
            5. Focus on providing factual, educational information
            
            Please provide your best answer:
            """
        else:
            prompt = f"""
            Please answer this question based on your own knowledge. Be concise and direct.
            
            Question: {question}
            
            IMPORTANT:
            1. Your answer must be strictly under {max_words} words
            2. Focus only on the most essential information
            3. If you're uncertain about specific facts, acknowledge the limitations of your knowledge
            
            Please provide your best answer:
            """
        
        try:
            logger.info(f"Generating direct answer for question: {question[:50]}... (no search results available)")
            return self.llm.generate(prompt).strip()
        except Exception as e:
            logger.error(f"Direct answer generation failed: {str(e)}")
            return f"Unable to generate an answer due to: {str(e)}"

class RoleCoordinator:
    """规划多智能体讨论的角色协调器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.base_roles = [
            "General Public Representative", 
            "Domain Expert Analyst", 
            "Critical Thinker"
        ]  # 基础角色池
        
    def analyze_topic(self, news_topic):
        """分析新闻主题，确定合适的讨论角色组合"""
        analysis_prompt = f"""
        Analyze the following news topic: "{news_topic}"
        
        1. What major domains does this news topic involve?
        2. Which groups of people are most concerned about or affected by this?
        3. What professional perspectives are needed for a comprehensive understanding?
        
        Based on the analysis above, select 3-5 roles from the following basic roles that are most suitable for discussing this topic:
        - General Public Representative
        - Domain Expert Analyst
        - Critical Thinker
        - Historical Perspective Analyst
        - Future Outlook Analyst
        - Ethics Considerations Analyst
        - Social Impact Evaluator
        - Economic Perspective Analyst
        - Policy Interpreter
        - Technology Perspective Analyst
        
        For each selected role, provide a brief role positioning statement including:
        1. Which aspects of the news this role focuses on
        2. What perspective this role represents
        
        IMPORTANT: Return ONLY the JSON format without any other text, exactly as follows:
        {{
        "roles": [
            {{"name": "Role Name", "description": "Role positioning statement", "focus": "Focus point"}}
        ]
        }}
        """
        
        response = self.llm.generate(analysis_prompt)
        try:
            # 清理响应文本以提取有效的JSON
            cleaned_response = self._clean_json_response(response)
            roles_data = json.loads(cleaned_response)
            return roles_data["roles"]
        except Exception as e:
            # 记录具体错误，便于调试
            logger.warning(f"Failed to parse role data: {str(e)}")
            logger.warning(f"Raw response: {response[:200]}...")
            
            # 尝试使用更简单的解析方式
            try:
                # 直接提取角色信息
                return self._extract_roles_from_text(response)
            except:
                # 最终后备方案
                logger.warning("Using default roles as fallback")
                return [
                    {"name": "General Public Representative", "description": "Represents the general public perspective", "focus": "Public concerns"},
                    {"name": "Domain Expert Analyst", "description": "Focuses on professional analysis", "focus": "Expert insights"},
                    {"name": "Critical Thinker", "description": "Provides a critical perspective", "focus": "Questioning and analysis"}
                ]
        
    def _clean_json_response(self, text):
        """清理响应文本，提取有效JSON"""
        # 去除前后空白
        text = text.strip()
        
        # 处理markdown代码块
        if "```json" in text:
            # 提取json代码块内容
            parts = text.split("```json")
            if len(parts) > 1:
                text = parts[1]
                if "```" in text:
                    text = text.split("```")[0]
        elif "```" in text:
            # 提取普通代码块内容
            parts = text.split("```")
            if len(parts) > 1:
                text = parts[1]
        
        # 处理常见的JSON格式问题
        text = text.strip()
        
        # 确保文本是有效的JSON
        if not (text.startswith('{') and text.endswith('}')):
            raise ValueError("Response is not valid JSON")
            
        return text

    def _extract_roles_from_text(self, text):
        """从文本中提取角色信息"""
        # 针对特定模式识别角色
        roles = []
        lines = text.split("\n")
        current_role = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 匹配角色名称模式
            if '"name":' in line or '"Name":' in line or 'name:' in line:
                name_parts = line.split(":", 1)
                if len(name_parts) > 1:
                    name = name_parts[1].strip().strip('",').strip()
                    current_role = {"name": name}
            
            # 匹配描述模式
            elif ('"description":' in line or '"Description":' in line or 'description:' in line) and current_role:
                desc_parts = line.split(":", 1)
                if len(desc_parts) > 1:
                    desc = desc_parts[1].strip().strip('",').strip()
                    current_role["description"] = desc
            
            # 匹配焦点模式
            elif ('"focus":' in line or '"Focus":' in line or 'focus:' in line) and current_role:
                focus_parts = line.split(":", 1)
                if len(focus_parts) > 1:
                    focus = focus_parts[1].strip().strip('",').strip()
                    current_role["focus"] = focus
                    
                    # 添加完整的角色并重置
                    if "name" in current_role and "description" in current_role and "focus" in current_role:
                        roles.append(current_role)
                        current_role = None
        
        # 如果提取到角色，返回
        if roles:
            return roles
        
        # 否则抛出异常
        raise ValueError("Could not extract roles from text")
    
    def generate_agent_prompts(self, roles, news_topic):
        """为每个角色生成提示词"""
        agent_prompts = {}
        
        for role in roles:
            prompt = f"""
            You are '{role['name']}', {role['description']}
            
            You are participating in a news discussion about "{news_topic}". As {role['name']}, you particularly focus on {role['focus']}.
            
            Please propose 3-5 of the most important questions about this news topic from your perspective. These questions should:
            1. Reflect your unique perspective and concerns
            2. Help deepen the understanding of the news topic
            3. Include a mix of open-ended and specific questions
            4. Each question must be concise (20-30 words maximum)
            
            List the questions directly without additional explanation.
            """
            agent_prompts[role['name']] = prompt
            
        return agent_prompts
            
    def orchestrate_discussion(self, news_topic):
        """协调多智能体讨论，生成问题集"""
        # 1. 分析主题，确定角色
        roles = self.analyze_topic(news_topic)
        
        # 2. 为每个角色生成提示词
        agent_prompts = self.generate_agent_prompts(roles, news_topic)
        
        # 3. 让每个角色生成问题
        all_questions = []
        for role_name, prompt in agent_prompts.items():
            logger.info(f"Generating questions for role '{role_name}'")
            role_questions = self.llm.generate(prompt).strip().split('\n')
            # 清理问题格式
            role_questions = [q.strip().replace('- ', '') for q in role_questions if q.strip()]
            # 添加角色标记
            role_questions = [(q, role_name) for q in role_questions]
            all_questions.extend(role_questions)
            
        return all_questions, roles


class QuestionNode:
    """支持引力层级的问题节点"""
    
    def __init__(self, question_text, source_role=None):
        self.question_text = question_text
        self.answer_text = None
        self.source_role = source_role
        self.mass = 1.0
        self.position = (0, 0)
        
        # 新增 - 引力层级属性
        self.gravity_level = None  # 'core', 'middle', 'peripheral'
        self.level_score = 0.0     # 层级得分
        
        # 节点属性
        self.properties = {
            'relevance': 0.0,
            'controversy': 0.0,
            'informativeness': 0.0,
            'abstractness': 0.0,    # 新增 - 抽象度
            'connectivity': 0.0     # 新增 - 连接性
        }
        
        # 信息来源
        self.related_sources = []
        
        # 观点相关
        self.current_opinion = None  # 当前观点
        self.opinion_history = []    # 观点历史
        
        # 新增 - 引力路径相关
        self.gravity_paths = []      # 引力路径列表，每个路径是节点ID序列
        self.related_nodes = {}      # 相关节点 {node_id: similarity_score}
        
    def set_opinion(self, opinion):
        """设置节点的当前观点"""
        self.current_opinion = opinion
        
        # 记录到历史
        self.opinion_history.append({
            "timestamp": time.time(),
            "opinion": opinion
        })
        
    def get_opinion_text(self):
        """获取当前观点文本，如果存在的话"""
        if self.current_opinion and "text" in self.current_opinion:
            return self.current_opinion["text"]
        return None
        
    def has_opinion(self):
        """检查是否有观点"""
        return self.current_opinion is not None
        
    def set_gravity_level(self, level, score=0.0):
        """设置节点的引力层级
        
        参数:
            level: 引力层级 ('core', 'middle', 'peripheral')
            score: 层级得分
        """
        self.gravity_level = level
        self.level_score = score
        
    def is_peripheral(self):
        """判断节点是否为外围层节点（用于观点控制）"""
        return self.gravity_level == 'peripheral'
        
    def add_gravity_path(self, path):
        """添加引力路径
        
        参数:
            path: 节点ID序列，从核心到当前节点
        """
        if path not in self.gravity_paths:
            self.gravity_paths.append(path)


class EnhancedOpinionGenerator:
    """Opinion generator with sentiment control utilizing gravity field edge relationships"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def generate_opinion_for_node(self, gravity_field, node_id, sentiment_control=None):
        """Generate sentiment-controlled opinion for a single node, utilizing edge relationships"""
        if node_id not in gravity_field.nodes:
            return None
            
        target_node = gravity_field.nodes[node_id]
        if not target_node.answer_text:
            return None
            
        # 1. Collect related node information
        related_info = self._collect_related_nodes_info(gravity_field, node_id)
        
        # 2. Determine sentiment control instruction
        sentiment_instruction = self._get_sentiment_instruction(sentiment_control)
        
        # 3. Generate enhanced opinion
        prompt = self._create_enhanced_opinion_prompt(
            target_node, 
            related_info, 
            sentiment_instruction
        )
        
        logger.info(f"Generating opinion for question: {target_node.question_text[:30]}...")
        opinion_text = self.llm.generate(prompt).strip()
        
        # 4. Analyze sentiment of generated opinion
        actual_sentiment = self._analyze_sentiment(opinion_text)
        
        # 5. Construct opinion object
        opinion = {
            "text": opinion_text,
            "intended_sentiment": sentiment_control,
            "actual_sentiment": actual_sentiment,
            "supporting_nodes": [info["node_id"] for info in related_info],
            "generation_timestamp": time.time()
        }
        
        return opinion
    
    def _collect_related_nodes_info(self, gravity_field, node_id):
        """Collect information about nodes related to the target node"""
        related_info = []
        target_node = gravity_field.nodes[node_id]
        
        # Find all edges connected to this node
        connected_edges = []
        for edge in gravity_field.edges:
            id1, id2, force = edge
            if id1 == node_id or id2 == node_id:
                connected_edges.append(edge)
        
        # Sort by gravitational force
        connected_edges.sort(key=lambda x: x[2], reverse=True)
        
        # Select the top 3 most strongly related nodes
        for edge in connected_edges[:3]:
            id1, id2, force = edge
            related_id = id2 if id1 == node_id else id1
            related_node = gravity_field.nodes[related_id]
            
            # Only use related nodes that have answers
            if related_node.answer_text:
                related_info.append({
                    "node_id": related_id,
                    "question": related_node.question_text,
                    "answer": related_node.answer_text,
                    "relation_strength": force,
                    # Determine relationship type (similar/complementary/contrasting)
                    "relation_type": self._determine_relation_type(target_node, related_node, force)
                })
        
        return related_info
    
    def _determine_relation_type(self, target_node, related_node, force):
        """Determine the relationship type between two nodes"""
        # Compare target question with related question to determine relationship
        if force > 8:  # Strong connection
            return "Supporting relationship"
        elif force > 5:  # Medium connection
            return "Complementary relationship"
        else:  # Weak connection
            return "Background relationship"
    
    def _get_sentiment_instruction(self, sentiment_control):
        """Construct instruction based on sentiment control value"""
        if sentiment_control is None:
            return "maintain an objective and balanced attitude"
            
        if isinstance(sentiment_control, str):
            # Handle text-based sentiment control
            sentiment_map = {
                "positive": "positive and optimistic",
                "negative": "critical and cautious",
                "neutral": "completely neutral",
                "balanced": "objective and balanced"
            }
            return f"use a {sentiment_map.get(sentiment_control, 'objective and balanced')} attitude"
        
        # Handle numeric sentiment control (-1.0 to 1.0)
        if sentiment_control > 0.3:
            return f"use a positive, optimistic attitude with an intensity of {sentiment_control:.1f} (on a 0-1 scale)"
        elif sentiment_control < -0.3:
            return f"use a cautious, critical attitude with an intensity of {abs(sentiment_control):.1f} (on a 0-1 scale)"
        else:
            return "use a neutral, objective attitude"
    
    def _create_enhanced_opinion_prompt(self, target_node, related_info, sentiment_instruction):
        """Create enhanced opinion prompt utilizing edge relationships"""
        prompt = f"""
        Please generate an opinion based on the following main question and answer:
        
        Main Question: {target_node.question_text}
        
        Main Answer: {target_node.answer_text[:300]}...
        
        Sentiment Requirement: {sentiment_instruction}
        """
        
        # Add related question information (限制数量)
        if related_info:
            # 限制最多使用2个相关信息
            limited_info = related_info[:2]
            prompt += "\n\nRelated Information (for enhancing and supporting the opinion):"
            
            for i, info in enumerate(limited_info, 1):
                prompt += f"""
                
                Related Question {i} ({info['relation_type']}):
                Question: {info['question']}
                Answer: {info['answer'][:150]}...
                """
        
        prompt += f"""
        
        Please integrate the main question and related information to generate a deep, persuasive opinion that:
        1. Directly expresses a clear stance and judgment
        2. Uses related information to support or enhance the opinion
        3. {sentiment_instruction}
        4. Maintains logical coherence
        5. STRICTLY LIMIT your response to 100-150 words only
        
        Return only the opinion text without any other explanation.
        """
        
        return prompt
    
    def _analyze_sentiment(self, text):
        """Analyze the sentiment of text, returning a value from -1 to 1"""
        # Simplified sentiment analysis
        prompt = f"""
        Analyze the sentiment of the following text, returning a value between -1 and 1:
        -1 represents extremely negative
        0 represents neutral
        1 represents extremely positive
        
        Text: {text}
        
        Return only the numeric value, without any explanation.
        """
        
        try:
            response = self.llm.generate(prompt).strip()
            return float(response)
        except:
            # Default to neutral value
            logger.warning(f"Sentiment analysis failed, returning default value 0.0")
            return 0.0
MODEL_PATH = "../models/paraphrase-multilingual-MiniLM-L12-v2"
class NewsGravityField:
    """基于引力层级的新闻引力场"""
    
    def __init__(self, topic, encoder=None, llm=None):
        self.topic = topic
        
        # 初始化编码器
        if encoder:
            self.encoder = encoder
        else:
            try:
                self.encoder = SentenceTransformer(MODEL_PATH)
                logger.info("成功加载编码器")
            except Exception as e:
                logger.error(f"加载编码器失败: {e}")
                raise

        # 核心数据结构
        self.nodes = {}               # 问题节点字典 {question_id: QuestionNode}
        self.connections = []         # 节点连接 [(node_id1, node_id2, strength)]
        self.center_node_id = None    # 中心节点ID
        
        # 编码缓存
        self.embeddings = {}          # 问题文本的嵌入向量
        
        # 层级节点组
        self.core_nodes = []          # 核心层节点
        self.middle_nodes = []        # 中间层节点
        self.peripheral_nodes = []    # 外围层节点（观点控制层）
        
        # 观点生成器
        self.llm = llm
        if llm:
            self.opinion_generator = EnhancedOpinionGenerator(llm)
        else:
            self.opinion_generator = None
        
        logger.info(f"初始化引力层级场: {topic}")

    def params_to_text(self, params):
        """将数值参数转换为自然语言描述"""
        # 提取参数
        bias = params.get('bias', 0.0)
        emotion = params.get('emotion', 0.0) 
        exaggeration = params.get('exaggeration', 0.3)
        
        # 构造情感描述
        if emotion < -0.6:
            emotion_desc = "very negative and critical"
        elif emotion < -0.2:
            emotion_desc = "negative and critical"
        elif emotion < 0.2:
            emotion_desc = "neutral and objective"
        elif emotion < 0.6:
            emotion_desc = "positive and supportive"
        else:
            emotion_desc = "very positive and enthusiastically supportive"
        
        # 构造偏见描述
        if bias > 0.7:
            bias_desc = "strongly supportive and actively approving"
        elif bias > 0.3:
            bias_desc = "supportive and approving"
        elif bias > -0.3:
            bias_desc = "relatively neutral"
        elif bias > -0.7:
            bias_desc = "opposing and disapproving"
        else:
            bias_desc = "strongly opposing and firmly rejecting"
        
        # 构造夸张描述
        if exaggeration > 0.7:
            exaggeration_desc = ", significantly exaggerating facts and impacts"
        elif exaggeration > 0.4:
            exaggeration_desc = ", moderately exaggerating some facts"
        else:
            exaggeration_desc = ""
            
        return emotion_desc, bias_desc, exaggeration_desc
        
    def add_question(self, question_text, source_role=None):
        """添加问题到引力场"""
        question_id = hashlib.md5(question_text.encode('utf-8')).hexdigest()
        
        # 检查是否已存在
        if question_id in self.nodes:
            logger.info(f"问题已存在: {question_text[:30]}...")
            return question_id
        
        # 创建新节点
        node = QuestionNode(question_text, source_role)
        self.nodes[question_id] = node
        
        # 计算并缓存嵌入向量
        embedding = self.encoder.encode([question_text])[0]
        self.embeddings[question_id] = embedding
        
        # 如果是第一个节点，设为中心
        if len(self.nodes) == 1:
            self.center_node_id = question_id
        
        logger.info(f"添加问题: {question_text[:30]}...")
        return question_id
    
    def add_answer(self, question_id, answer_text):
        """为问题添加答案"""
        if question_id in self.nodes:
            self.nodes[question_id].answer_text = answer_text
            
            # 重新计算节点质量
            self.calculate_node_mass(question_id)
            
            logger.info(f"添加答案到问题: {self.nodes[question_id].question_text[:30]}...")
            return True
        else:
            logger.warning(f"节点不存在: {question_id}")
            return False
    
    def calculate_node_mass(self, question_id):
        """计算节点的引力质量"""
        if question_id not in self.nodes:
            logger.warning(f"计算质量: 节点不存在 {question_id}")
            return 0.0
            
        node = self.nodes[question_id]
        
        # 基础质量为1.0
        base_mass = 1.0
        
        # 如果有答案，基于答案质量调整
        if node.answer_text:
            # 答案长度因子(适当长度的答案质量更高)
            length = len(node.answer_text)
            length_factor = min(1.0, length/300) if length < 300 else min(2.0, 600/length)
            
            # 计算答案与主题的相关性
            answer_embedding = self.encoder.encode([node.answer_text])[0]
            topic_embedding = self.encoder.encode([self.topic])[0]
            relevance = cosine_similarity([answer_embedding], [topic_embedding])[0][0]
            
            # 保存相关性属性
            node.properties['relevance'] = relevance
            
            # 综合计算质量
            mass = base_mass * (0.5 + length_factor) * (0.5 + relevance)#参数
            
        else:
            mass = base_mass
            
        node.mass = mass
        return mass
    
    def build_gravity_connections(self):
        """构建节点之间的引力连接（基于相似度）"""
        # 清空现有连接
        self.connections = []
        
        # 获取所有节点ID
        node_ids = list(self.nodes.keys())
        
        if len(node_ids) < 2:
            logger.warning("构建引力连接: 节点数量不足")
            return []
            
        logger.info(f"构建引力连接: {len(node_ids)}个节点")
        
        # 计算所有可能的节点对之间的引力
        for i, id1 in enumerate(node_ids):
            for j, id2 in enumerate(node_ids):
                if i >= j:  # 避免重复计算
                    continue
                    
                # 计算语义相似度
                similarity = cosine_similarity(
                    [self.embeddings[id1]], 
                    [self.embeddings[id2]]
                )[0][0]
                
                # 计算质量乘积
                mass_product = self.nodes[id1].mass * self.nodes[id2].mass
                
                # 计算引力强度: F = G * (m1*m2)/r²
                distance = max(0.1, 1.0 - similarity)  # 防止除零
                strength = 10 * mass_product / (distance * distance)#G引力值
                
                # 只添加高于阈值的连接
                if strength > 3.0:  # 最小引力阈值，建立关系的引力阈值
                    self.connections.append((id1, id2, strength))
                    
                    # 为节点添加相关节点信息
                    self.nodes[id1].related_nodes[id2] = similarity
                    self.nodes[id2].related_nodes[id1] = similarity
        
        # 计算每个节点的连接性属性
        for node_id, node in self.nodes.items():
            connectivity = sum(node.related_nodes.values())
            node.properties['connectivity'] = connectivity
        
        logger.info(f"构建了 {len(self.connections)} 个引力连接")
        
        # 根据连接结构进行层级分类
        self.classify_node_levels()
        
        # 计算引力路径
        self.calculate_gravity_paths()
        
        return self.connections
    
    def classify_node_levels(self):
        """将节点分类为核心层、中间层和外围层"""
        # 清空现有层级分组
        self.core_nodes = []
        self.middle_nodes = []
        self.peripheral_nodes = []
        
        # 如果节点数量太少，无法有效分层
        if len(self.nodes) < 3:
            logger.warning("节点数量不足，无法进行有效分层")
            return
        
        if self.llm:
            # 使用大模型进行层级分类
            self._classify_with_llm()
        else:
            # 使用启发式方法进行层级分类
            self._classify_heuristically()
        
        # 记录分类统计信息
        logger.info(f"节点层级分类完成: 核心层 {len(self.core_nodes)} 节点，"
                   f"中间层 {len(self.middle_nodes)} 节点，"
                   f"外围层 {len(self.peripheral_nodes)} 节点")
    
    def _classify_heuristically(self):
        """使用启发式方法进行节点层级分类
        
        基于节点的以下特性:
        1. 连接性 - 与其他节点的连接强度
        2. 主题中心度 - 与主题的相似度
        3. 节点质量 - 基于答案质量和相关性
        """
        # 为每个节点计算层级得分
        scores = {}
        
        for node_id, node in self.nodes.items():
            # 计算连接强度总和
            connection_strength = 0
            connection_count = 0
            
            for conn in self.connections:
                id1, id2, strength = conn
                if id1 == node_id or id2 == node_id:
                    connection_strength += strength
                    connection_count += 1
            
            # 计算与主题的相似度
            topic_embedding = self.encoder.encode([self.topic])[0]
            node_embedding = self.embeddings[node_id]
            topic_similarity = cosine_similarity([node_embedding], [topic_embedding])[0][0]
            
            # 计算抽象度 (反比于问题长度)
            question_length = len(node.question_text)
            abstractness = 1.0 - min(1.0, question_length / 50)
            node.properties['abstractness'] = abstractness
            
            # 综合计算层级得分
            # 高得分 = 核心层，低得分 = 外围层
            level_score = (
                0.3 * connection_strength +
                0.3 * (connection_count if connection_count > 0 else 0) +
                0.2 * topic_similarity +
                0.1 * node.mass +
                0.1 * abstractness
            )
            
            scores[node_id] = level_score
            node.level_score = level_score
        
        # 基于得分进行分类
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        total_nodes = len(sorted_nodes)
        
        # 核心层: 得分最高的20%
        core_count = max(1, int(total_nodes * 0.2))
        # 外围层: 得分最低的35%
        peripheral_count = max(1, int(total_nodes * 0.35))
        # 中间层: 其他节点
        
        # 分配层级
        for i, (node_id, score) in enumerate(sorted_nodes):
            if i < core_count:
                self.nodes[node_id].set_gravity_level('core', score)
                self.core_nodes.append(node_id)
            elif i >= total_nodes - peripheral_count:
                self.nodes[node_id].set_gravity_level('peripheral', score)
                self.peripheral_nodes.append(node_id)
            else:
                self.nodes[node_id].set_gravity_level('middle', score)
                self.middle_nodes.append(node_id)
    
    def _classify_with_llm(self):
        """使用大模型进行节点层级分类"""
        # 构建节点信息
        nodes_info = []
        
        for node_id, node in self.nodes.items():
            # 计算与主题的相似度
            topic_embedding = self.encoder.encode([self.topic])[0]
            node_embedding = self.embeddings[node_id]
            topic_similarity = round(cosine_similarity([node_embedding], [topic_embedding])[0][0], 2)
            
            # 计算连接数量
            connection_count = 0
            for id1, id2, _ in self.connections:
                if id1 == node_id or id2 == node_id:
                    connection_count += 1
            
            # 构建节点信息
            info = {
                "id": node_id[:8],  # 使用ID的前8位
                "question": node.question_text,
                "has_answer": bool(node.answer_text),
                "connections": connection_count,
                "topic_similarity": topic_similarity
            }
            nodes_info.append(info)
        
        prompt = f"""
        You are a knowledge structure expert responsible for classifying question nodes into different gravity levels.

        Topic: {self.topic}

        The nodes need to be classified into three levels:
        1. Core Level: The most fundamental, abstract, and central questions, forming the core of the gravity field
        2. Middle Level: Questions connecting the core and periphery, serving as transmission links
        3. Peripheral Level: The most specific, application-oriented questions, where opinions are formed

        Node information:
        """

        # Add node information
        for i, info in enumerate(nodes_info, 1):
            prompt += f"""
            Node {i}: {info['id']}
            Question: {info['question']}
            Has answer: {'Yes' if info['has_answer'] else 'No'}
            Connection count: {info['connections']}
            Topic similarity: {info['topic_similarity']}
            """

        # Add classification request
        prompt += """
        Please analyze these nodes and classify them into core, middle, and peripheral levels. Consider the following factors:
        - The abstraction/specificity level of the question
        - Relevance to the main topic
        - Number of connections
        - Whether it's a basic concept or an application scenario

        Return the classification results in JSON format:
        {
        "core": ["NodeID1", "NodeID2", ...],
        "middle": ["NodeID3", "NodeID4", ...],
        "peripheral": ["NodeID5", "NodeID6", ...]
        }

        Ensure:
        1. Every node is assigned to a level
        2. Core level nodes comprise approximately 20%
        3. Peripheral level nodes comprise approximately 35%
        4. Middle level nodes comprise approximately 45%

        Return only the JSON format, no additional explanation.
        """
        
        try:
            # 调用大模型
            response = self.llm.generate(prompt)
            
            # 解析JSON结果
            import json
            import re
            
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                classification = json.loads(json_str)
                
                # 处理分类结果
                core_ids = classification.get("core", classification.get("核心", []))
                middle_ids = classification.get("middle", classification.get("中间", []))
                peripheral_ids = classification.get("peripheral", classification.get("外围", []))
                
                # 将短ID映射到完整ID
                id_mapping = {node_id[:8]: node_id for node_id in self.nodes.keys()}
                
                # 更新节点层级
                for short_id in core_ids:
                    full_id = id_mapping.get(short_id)
                    if full_id:
                        self.nodes[full_id].set_gravity_level('core')
                        self.core_nodes.append(full_id)
                
                for short_id in middle_ids:
                    full_id = id_mapping.get(short_id)
                    if full_id:
                        self.nodes[full_id].set_gravity_level('middle')
                        self.middle_nodes.append(full_id)
                
                for short_id in peripheral_ids:
                    full_id = id_mapping.get(short_id)
                    if full_id:
                        self.nodes[full_id].set_gravity_level('peripheral')
                        self.peripheral_nodes.append(full_id)
                        
                logger.info("使用大模型成功完成节点层级分类")
            else:
                logger.error("无法解析大模型返回的分类结果，回退到启发式方法")
                self._classify_heuristically()
                
        except Exception as e:
            logger.error(f"使用大模型分类节点层级失败: {str(e)}")
            # 回退到启发式方法
            self._classify_heuristically()
    
    def calculate_gravity_paths(self):
        """计算引力路径 - 从核心层到外围层的信息流路径"""
        # 如果层级分类未完成，先进行分类
        if not self.core_nodes or not self.peripheral_nodes:
            self.classify_node_levels()
            
        # 为每个外围节点计算引力路径
        for peripheral_id in self.peripheral_nodes:
            # 获取外围节点连接的中间节点
            connected_nodes = []
            for id1, id2, strength in self.connections:
                if id1 == peripheral_id and id2 in self.middle_nodes:
                    connected_nodes.append((id2, strength))
                elif id2 == peripheral_id and id1 in self.middle_nodes:
                    connected_nodes.append((id1, strength))
            
            # 按连接强度排序
            connected_nodes.sort(key=lambda x: x[1], reverse=True)
            
            # 为每个连接的中间节点，找到通往核心节点的路径
            paths = []
            for middle_id, _ in connected_nodes[:2]:  # 限制使用的中间节点数量
                for core_id in self.core_nodes:
                    # 寻找从中间节点到核心节点的连接
                    core_connected = False
                    for id1, id2, _ in self.connections:
                        if (id1 == middle_id and id2 == core_id) or (id2 == middle_id and id1 == core_id):
                            core_connected = True
                            break
                    
                    if core_connected:
                        # 构建路径: 核心 -> 中间 -> 外围
                        path = [core_id, middle_id, peripheral_id]
                        paths.append(path)
            
            # 如果没有找到完整路径，尝试直接连接到核心节点
            if not paths:
                for core_id in self.core_nodes:
                    # 检查是否直接连接
                    directly_connected = False
                    for id1, id2, _ in self.connections:
                        if (id1 == peripheral_id and id2 == core_id) or (id2 == peripheral_id and id1 == core_id):
                            directly_connected = True
                            break
                    
                    if directly_connected:
                        # 构建直接路径: 核心 -> 外围
                        path = [core_id, peripheral_id]
                        paths.append(path)
            
            # 为外围节点保存路径
            for path in paths:
                self.nodes[peripheral_id].add_gravity_path(path)
            
            # 如果仍然没有路径，使用最强连接创建路径
            if not self.nodes[peripheral_id].gravity_paths:
                # 找出与该节点连接最强的节点
                strongest_connections = []
                for id1, id2, strength in self.connections:
                    if id1 == peripheral_id:
                        strongest_connections.append((id2, strength))
                    elif id2 == peripheral_id:
                        strongest_connections.append((id1, strength))
                
                if strongest_connections:
                    strongest_connections.sort(key=lambda x: x[1], reverse=True)
                    connected_id = strongest_connections[0][0]
                    path = [connected_id, peripheral_id]
                    self.nodes[peripheral_id].add_gravity_path(path)
        
        # 记录路径统计信息
        total_paths = sum(len(node.gravity_paths) for node in self.nodes.values())
        logger.info(f"计算了 {total_paths} 条引力路径，平均每个外围节点 {total_paths/max(1, len(self.peripheral_nodes)):.1f} 条路径")
    
    # def generate_peripheral_opinions(self, sentiment_controls=None, use_path_awareness=True):
    #     """为外围层节点生成可控观点
        
    #     参数:
    #         sentiment_controls: 字典 {node_id: sentiment_value} 或 None
    #         use_path_awareness: 是否使用路径感知生成观点
            
    #     返回:
    #         生成的观点数量
    #     """
    #     # 确保层级分类已完成
    #     if not self.peripheral_nodes:
    #         self.classify_node_levels()
            
    #     if not self.peripheral_nodes:
    #         logger.warning("没有外围层节点，无法生成观点")
    #         return 0
            
    #     # 初始化情感控制
    #     if sentiment_controls is None:
    #         sentiment_controls = {}
            
    #     logger.info(f"为 {len(self.peripheral_nodes)} 个外围层节点生成可控观点")
        
    #     # 记录成功生成的观点数量
    #     generated_count = 0
        
    #     for node_id in self.peripheral_nodes:
    #         # 获取节点
    #         node = self.nodes[node_id]
            
    #         # 跳过已有观点的节点
    #         if node.has_opinion():
    #             continue
                
    #         try:
    #             # 确定情感控制
    #             sentiment = sentiment_controls.get(node_id, "balanced")
                
    #             if use_path_awareness and node.gravity_paths:
    #                 # 使用路径感知的观点生成
    #                 success = self._generate_path_aware_opinion(node_id, sentiment)
    #             else:
    #                 # 使用普通的观点生成
    #                 success = self._generate_simple_opinion(node_id, sentiment)
                
    #             if success:
    #                 generated_count += 1
    #                 logger.info(f"成功生成节点 {node_id} 的观点")
                    
    #         except Exception as e:
    #             logger.error(f"处理节点 {node_id} 时出错: {str(e)}")
    #             continue
                        
    #     logger.info(f"成功为 {generated_count} 个外围层节点生成观点")
    #     return generated_count

    def generate_peripheral_opinions(self, sentiment_controls=None, use_path_awareness=True):
        """为外围层节点生成可控观点
        
        参数:
            sentiment_controls: 字典 {node_id: {bias, emotion, exaggeration}} 或 None
            use_path_awareness: 是否使用路径感知生成观点
                
        返回:
            生成的观点数量
        """
        # 确保层级分类已完成
        if not self.peripheral_nodes:
            self.classify_node_levels()
            
        if not self.peripheral_nodes:
            logger.warning("没有外围层节点，无法生成观点")
            return 0
            
        # 初始化情感控制
        if sentiment_controls is None:
            sentiment_controls = {}
            
        logger.info(f"为 {len(self.peripheral_nodes)} 个外围层节点生成可控观点")
        
        # 记录成功生成的观点数量
        generated_count = 0
        
        for node_id in self.peripheral_nodes:
            # 获取节点
            node = self.nodes[node_id]
            
            # 跳过已有观点的节点
            if node.has_opinion():
                continue
                
            try:
                # 获取节点的参数控制
                params = sentiment_controls.get(node_id, {"bias": 0.0, "emotion": 0.0, "exaggeration": 0.3})
                
                if use_path_awareness and node.gravity_paths:
                    # 使用路径感知的观点生成
                    success = self._generate_path_aware_opinion(node_id, params)
                else:
                    # 使用普通的观点生成
                    success = self._generate_simple_opinion(node_id, params)
                
                if success:
                    generated_count += 1
                    logger.info(f"成功生成节点 {node_id} 的观点")
                    
            except Exception as e:
                logger.error(f"处理节点 {node_id} 时出错: {str(e)}")
                continue
                        
        logger.info(f"成功为 {generated_count} 个外围层节点生成观点")
        return generated_count
    
    # def _generate_simple_opinion(self, node_id, sentiment_control):
    #     """生成简单的节点观点（不使用路径信息）
        
    #     参数:
    #         node_id: 节点ID
    #         sentiment_control: 情感控制值
            
    #     返回:
    #         bool: 是否成功生成观点
    #     """
    #     if node_id not in self.nodes:
    #         logger.warning(f"节点不存在: {node_id}")
    #         return False
            
    #     node = self.nodes[node_id]
        
    #     # 如果节点没有答案，无法生成观点
    #     if not node.answer_text:
    #         logger.warning(f"节点 {node_id} 没有答案，无法生成观点")
    #         return False
        
    #     # 确定情感控制指令
    #     sentiment_instruction = self._get_sentiment_instruction(sentiment_control)
        
    #     # 收集节点的相关信息
    #     related_info = []
    #     for related_id, similarity in list(node.related_nodes.items())[:2]:  # 只使用最相关的两个节点
    #         related_node = self.nodes.get(related_id)
    #         if related_node and related_node.answer_text:
    #             related_info.append({
    #                 "node_id": related_id,
    #                 "question": related_node.question_text,
    #                 "answer": related_node.answer_text[:200],  # 限制长度
    #                 "relation_type": f"Related Node (Similarity: {similarity:.2f})"
    #             })
        
    #     # 使用增强的观点生成器
    #     if self.opinion_generator:
    #         opinion = self.opinion_generator.generate_opinion_for_node(
    #             self, node_id, sentiment_control
    #         )
            
    #         if opinion:
    #             node.set_opinion(opinion)
    #             return True
                
    #     # 如果上面的方法失败，使用基本的观点生成
    #     prompt = self._create_opinion_prompt(node, related_info, sentiment_instruction)
        
    #     try:
    #         opinion_text = self.llm.generate(prompt).strip()
            
    #         # 构建观点对象
    #         opinion = {
    #             "text": opinion_text,
    #             "intended_sentiment": sentiment_control,
    #             "is_peripheral": True,
    #             "path_aware": False,
    #             "generation_timestamp": time.time()
    #         }
            
    #         # 设置观点
    #         node.set_opinion(opinion)
            
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"生成观点失败: {str(e)}")
    #         return False

    def _generate_simple_opinion(self, node_id, params):
        """生成简单的节点观点（不使用路径信息）
        
        参数:
            node_id: 节点ID
            params: 参数字典 {bias, emotion, exaggeration}
            
        返回:
            bool: 是否成功生成观点
        """
        if node_id not in self.nodes:
            logger.warning(f"节点不存在: {node_id}")
            return False
            
        node = self.nodes[node_id]
        
        # 如果节点没有答案，无法生成观点
        if not node.answer_text:
            logger.warning(f"节点 {node_id} 没有答案，无法生成观点")
            return False
        
        # 将参数转换为自然语言描述
        emotion_desc, bias_desc, exaggeration_desc = self.params_to_text(params)
        
        # 构建情感控制指令
        sentiment_instruction = f"use a {emotion_desc} tone, expressing a {bias_desc} stance{exaggeration_desc}"
        
        # 收集节点的相关信息
        related_info = []
        for related_id, similarity in list(node.related_nodes.items())[:2]:  # 只使用最相关的两个节点
            related_node = self.nodes.get(related_id)
            if related_node and related_node.answer_text:
                related_info.append({
                    "node_id": related_id,
                    "question": related_node.question_text,
                    "answer": related_node.answer_text[:200],  # 限制长度
                    "relation_type": f"Related Node (Similarity: {similarity:.2f})"
                })
        
        # 创建观点生成提示词
        prompt = self._create_opinion_prompt(node, related_info, sentiment_instruction)
        
        try:
            opinion_text = self.llm.generate(prompt).strip()
            
            # 构建观点对象
            opinion = {
                "text": opinion_text,
                "params": params,
                "is_peripheral": True,
                "path_aware": False,
                "generation_timestamp": time.time()
            }
            
            # 设置观点
            node.set_opinion(opinion)
            
            return True
            
        except Exception as e:
            logger.error(f"生成观点失败: {str(e)}")
            return False
      
    def _create_opinion_prompt(self, node, related_info, sentiment_instruction):
        """Create opinion generation prompt"""
        prompt = f"""
        Please generate an insightful opinion for the following question and answer:
        
        Question: {node.question_text}
        
        Answer: {node.answer_text[:300]}...
        
        Sentiment requirement: {sentiment_instruction}
        """
        
        # Add related information
        if related_info:
            prompt += "\n\nRelated information:"
            
            for i, info in enumerate(related_info, 1):
                prompt += f"""
                
                Related information {i} ({info['relation_type']}):
                Question: {info['question']}
                Answer: {info['answer']}
                """
        
        # Complete the prompt
        prompt += f"""
        
        Please generate a concise and powerful opinion that:
        1. Directly expresses a clear stance and judgment
        2. Integrates related information to support the opinion
        3. {sentiment_instruction}
        4. Maintains logical consistency
        5. Is strictly limited to 100-150 words
        
        Return only the opinion text, without any other explanation.
        """
        
        return prompt
    def _get_sentiment_instruction(self, sentiment_control):
        """确定情感控制指令
        
        参数:
            sentiment_control: 情感控制值或字符串
            
        返回:
            情感控制指令文本
        """
        if sentiment_control is None:
            return "maintain an objective and balanced attitude"
            
        if isinstance(sentiment_control, str):
            # 处理基于文本的情感控制
            sentiment_map = {
                "positive": "positive and optimistic",
                "negative": "critical and cautious",
                "neutral": "completely neutral",
                "balanced": "objective and balanced"
            }
            return f"use a {sentiment_map.get(sentiment_control, 'objective and balanced')} attitude"
        
        # 处理数值型情感控制 (-1.0 到 1.0)
        if sentiment_control > 0.3:
            return f"use a positive, optimistic attitude with an intensity of {sentiment_control:.1f} (on a 0-1 scale)"
        elif sentiment_control < -0.3:
            return f"use a cautious, critical attitude with an intensity of {abs(sentiment_control):.1f} (on a 0-1 scale)"
        else:
            return "use a neutral, objective attitude"
        

    # def _generate_path_aware_opinion(self, node_id, sentiment_control):
    #     """Generate a path-aware opinion
        
    #     Args:
    #         node_id: Node ID
    #         sentiment_control: Sentiment control value
            
    #     Returns:
    #         bool: Whether the opinion was successfully generated
    #     """
    #     if node_id not in self.nodes:
    #         logger.warning(f"Node does not exist: {node_id}")
    #         return False
            
    #     node = self.nodes[node_id]
        
    #     # If the node has no answer, opinion cannot be generated
    #     if not node.answer_text:
    #         logger.warning(f"Node {node_id} has no answer, cannot generate opinion")
    #         return False
            
    #     # If there is no path information, use simple opinion generation
    #     if not node.gravity_paths:
    #         logger.info(f"Node {node_id} has no gravity paths, using simple opinion generation")
    #         return self._generate_simple_opinion(node_id, sentiment_control)
        
    #     # Get the longest gravity path
    #     path = max(node.gravity_paths, key=len)
        
    #     # Collect information on nodes along the path
    #     path_context = []
        
    #     for i, path_node_id in enumerate(path):
    #         path_node = self.nodes.get(path_node_id)
    #         if not path_node or path_node_id == node_id:
    #             continue
                
    #         # Determine node level label
    #         level_label = "Core Level" if path_node.gravity_level == 'core' else \
    #                     "Middle Level" if path_node.gravity_level == 'middle' else \
    #                     "Peripheral Level"
            
    #         # Only use nodes with answers
    #         if path_node.answer_text:
    #             context_item = {
    #                 "node_id": path_node_id,
    #                 "question": path_node.question_text,
    #                 "answer": path_node.answer_text[:200],  # Limit answer length
    #                 "level": level_label,
    #                 "position": i  # Position in the path
    #             }
    #             path_context.append(context_item)
        
    #     # Determine sentiment control instruction
    #     sentiment_instruction = self._get_sentiment_instruction(sentiment_control)
        
    #     # Create path-aware prompt
    #     prompt = f"""
    #     You need to generate an opinion for a peripheral node in a knowledge gravity field. Please consider information along the gravity path from core to periphery.
        
    #     Topic: {self.topic}
        
    #     Peripheral Node Question: {node.question_text}
    #     Peripheral Node Answer: {node.answer_text[:300]}...
    #     """
        
    #     # Add path information
    #     if path_context:
    #         prompt += "\n\nGravity Path Information (from core to periphery):"
            
    #         for i, context in enumerate(path_context):
    #             prompt += f"""
                
    #             Path Node {i+1} ({context['level']}):
    #             Question: {context['question']}
    #             Answer Summary: {context['answer']}
    #             """
        
    #     # Complete the prompt
    #     prompt += f"""
        
    #     Please generate an insightful opinion that:
    #     1. Directly expresses a clear stance and judgment on the peripheral node question
    #     2. Integrates information along the gravity path, demonstrating understanding of the overall knowledge flow
    #     3. {sentiment_instruction}
    #     4. Maintains logical coherence and flows naturally
    #     5. Is strictly limited to 100-150 words
        
    #     Return only the opinion text, without any additional explanation.
    #     """
        
    #     try:
    #         opinion_text = self.llm.generate(prompt).strip()
            
    #         # Construct opinion object
    #         opinion = {
    #             "text": opinion_text,
    #             "intended_sentiment": sentiment_control,
    #             "is_peripheral": True,
    #             "path_aware": True,
    #             "path_length": len(path),
    #             "generation_timestamp": time.time()
    #         }
            
    #         # Set the opinion
    #         node.set_opinion(opinion)
            
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"Failed to generate path-aware opinion: {str(e)}")
    #         return False
    def _generate_path_aware_opinion(self, node_id, params):
        """生成路径感知的观点
        
        参数:
            node_id: 节点ID
            params: 参数字典 {bias, emotion, exaggeration}
            
        返回:
            bool: 是否成功生成观点
        """
        if node_id not in self.nodes:
            logger.warning(f"节点不存在: {node_id}")
            return False
            
        node = self.nodes[node_id]
        
        # 如果节点没有答案，无法生成观点
        if not node.answer_text:
            logger.warning(f"节点 {node_id} 没有答案，无法生成观点")
            return False
            
        # 如果没有路径信息，使用简单观点生成
        if not node.gravity_paths:
            logger.info(f"节点 {node_id} 没有引力路径，使用简单观点生成")
            return self._generate_simple_opinion(node_id, params)
        
        # 获取最长引力路径
        path = max(node.gravity_paths, key=len)
        
        # 收集路径上节点的信息
        path_context = []
        
        for i, path_node_id in enumerate(path):
            path_node = self.nodes.get(path_node_id)
            if not path_node or path_node_id == node_id:
                continue
                
            # 确定节点层级标签
            level_label = "Core Level" if path_node.gravity_level == 'core' else \
                        "Middle Level" if path_node.gravity_level == 'middle' else \
                        "Peripheral Level"
            
            # 只使用有答案的节点
            if path_node.answer_text:
                context_item = {
                    "node_id": path_node_id,
                    "question": path_node.question_text,
                    "answer": path_node.answer_text[:200],  # 限制答案长度
                    "level": level_label,
                    "position": i  # 路径中的位置
                }
                path_context.append(context_item)
        
        # 将参数转换为自然语言描述
        emotion_desc, bias_desc, exaggeration_desc = self.params_to_text(params)
        
        # 构建情感控制指令
        sentiment_instruction = f"use a {emotion_desc} tone, expressing a {bias_desc} stance{exaggeration_desc}"
        
        # 创建路径感知提示词
        prompt = f"""
        You need to generate an opinion for a peripheral node in a knowledge gravity field. Please consider information along the gravity path from core to periphery.
        
        Topic: {self.topic}
        
        Peripheral Node Question: {node.question_text}
        Peripheral Node Answer: {node.answer_text[:300]}...
        """
        
        # 添加路径信息
        if path_context:
            prompt += "\n\nGravity Path Information (from core to periphery):"
            
            for i, context in enumerate(path_context):
                prompt += f"""
                
                Path Node {i+1} ({context['level']}):
                Question: {context['question']}
                Answer Summary: {context['answer']}
                """
        
        # 完成提示词
        prompt += f"""
        
        Please generate an insightful opinion that:
        1. Directly expresses a clear stance and judgment on the peripheral node question
        2. Integrates information along the gravity path, demonstrating understanding of the overall knowledge flow
        3. {sentiment_instruction}
        4. Maintains logical coherence and flows naturally
        5. Is strictly limited to 100-150 words
        
        Return only the opinion text, without any additional explanation.
        """
        
        try:
            opinion_text = self.llm.generate(prompt).strip()
            
            # 构建观点对象
            opinion = {
                "text": opinion_text,
                "params": params,
                "is_peripheral": True,
                "path_aware": True,
                "path_length": len(path),
                "generation_timestamp": time.time()
            }
            
            # 设置观点
            node.set_opinion(opinion)
            
            return True
            
        except Exception as e:
            logger.error(f"生成路径感知观点失败: {str(e)}")
            return False
            
    def expand_knowledge(self, max_new_questions=10, prioritize_peripherals=True):
        """扩展知识引力场，特别关注外围层
        
        参数:
            max_new_questions: 最大新增问题数量
            prioritize_peripherals: 是否优先扩展外围层节点
            
        返回:
            新增节点数量
        """
        # 确保层级分类已完成
        if not self.peripheral_nodes:
            self.classify_node_levels()
        
        # 筛选需要扩展的节点
        nodes_to_expand = []
        
        # 优先处理外围层节点
        if prioritize_peripherals and self.peripheral_nodes:
            nodes_to_expand.extend([
                {'node_id': node_id, 'type': 'peripheral'} 
                for node_id in self.peripheral_nodes
            ])
        
        # 如果外围节点不足，添加其他有答案的节点
        if len(nodes_to_expand) < max_new_questions // 2:
            answered_nodes = [
                {'node_id': node_id, 'type': node.gravity_level} 
                for node_id, node in self.nodes.items()
                if node.answer_text and node_id not in self.peripheral_nodes
            ]
            
            # 随机选择一些节点补充
            import random
            random.shuffle(answered_nodes)
            nodes_to_expand.extend(answered_nodes[:max_new_questions - len(nodes_to_expand)])
        
        logger.info(f"计划从 {len(nodes_to_expand)} 个节点扩展知识")
        
        # 扩展过程计数
        new_questions_count = 0
        
        # 对每个待扩展节点执行扩展
        for node_info in nodes_to_expand:
            node_id = node_info['node_id']
            node = self.nodes[node_id]
            
            # 跳过没有角色的节点
            if not node.source_role:
                continue
                
            # 获取节点的答案内容
            if not node.answer_text:
                continue
                
            # 根据节点类型选择扩展方式
            is_peripheral = node_info['type'] == 'peripheral'
            
            # 生成后续问题
            follow_up_questions = self._generate_follow_up_questions(
                node_id,
                node.source_role,
                is_peripheral=is_peripheral
            )
            
            # 对于每个后续问题，检查冗余性并添加到图中
            for question in follow_up_questions:
                # 检查是否已达到最大问题数量
                if new_questions_count >= max_new_questions:
                    break
                    
                # 检查问题冗余性
                if self._is_question_redundant(question):
                    logger.info(f"跳过冗余问题: {question[:30]}...")
                    continue
                    
                # 添加新问题，保留父节点的角色
                new_node_id = self.add_question(question, node.source_role)
                
                # 为新节点设置一个初始层级（将在后续的重新构建连接时更新）
                # 如果扩展自外围节点，新节点也标记为外围层；否则标记为其父节点的层级
                new_node_level = 'peripheral' if is_peripheral else node.gravity_level
                self.nodes[new_node_id].set_gravity_level(new_node_level)
                
                # 添加引力连接，基于语义相似度
                # 计算语义相似度
                similarity = cosine_similarity(
                    [self.embeddings[node_id]], 
                    [self.embeddings[new_node_id]]
                )[0][0]
                
                # 添加连接，强度基于相似度
                connection_strength = 5.0 * similarity  # 给予新连接一个合理的强度
                self.connections.append((node_id, new_node_id, connection_strength))
                
                # 记录相关节点信息
                self.nodes[node_id].related_nodes[new_node_id] = similarity
                self.nodes[new_node_id].related_nodes[node_id] = similarity
                
                new_questions_count += 1
                logger.info(f"添加新问题: {question[:50]}...")
        
        if new_questions_count > 0:
            logger.info(f"成功添加 {new_questions_count} 个新问题")
            
            # 重新构建引力连接和层级分类
            self.build_gravity_connections()
        else:
            logger.info("没有添加新问题")
        
        return new_questions_count
    def _generate_follow_up_questions(self, node_id, role, max_questions=2, is_peripheral=False):
        """生成后续问题，特别关注外围层节点
        
        参数:
            node_id: 节点ID
            role: 角色
            max_questions: 最大问题数
            is_peripheral: 是否是外围层节点
            
        返回:
            问题列表
        """
        node = self.nodes[node_id]
        
        # 构建提示词
        context = f"""
        Original Question: {node.question_text}
        Answer: {node.answer_text[:300]}...
        """
        
        # 如果有观点，也包含在上下文中
        if node.has_opinion():
            context += f"\nOpinion: {node.get_opinion_text()}"
        
        # 获取节点层级信息
        level_info = "外围层" if is_peripheral else (
            "核心层" if node.gravity_level == 'core' else
            "中间层" if node.gravity_level == 'middle' else
            "未分类层"
        )
        
        # 根据节点类型创建不同的提示词
        if is_peripheral:
            # 外围层节点的提示词，侧重于应用和细节
            prompt = f"""
            You are analyzing a peripheral layer node in the knowledge gravitational field about "{self.topic}".
            
            Peripheral layer nodes are the most concrete, application-oriented questions, closely related to real cases and specific applications.
            
            Please read the following information:
            {context}
            
            As '{role}', please generate {max_questions} follow-up questions that should:
            1. Explore concrete applications, examples, or case studies in more depth
            2. Focus on practical impacts, consequences, or effects
            3. Dig into more details or edge cases
            4. Propose more specific, professional extension questions
            
            Each question should be concise and clear, not exceeding 30 characters, directly related to expanding peripheral knowledge.
            
            Please return only a list of questions, one per line.
            """
        elif node.gravity_level == 'core':
            # 核心层节点的提示词，侧重于基础概念扩展
            prompt = f"""
            You are analyzing a core layer node in the knowledge gravitational field about "{self.topic}".
            
            Core layer nodes are the most basic, abstract, and central questions, forming the foundation of the entire knowledge structure.
            
            Please read the following information:
            {context}
            
            As '{role}', please generate {max_questions} follow-up questions that should:
            1. Expand the breadth of fundamental concepts, covering more related core knowledge
            2. Explore theoretical foundations or principles
            3. Focus on understanding this core concept from different perspectives
            4. Maintain an appropriate level of abstraction, not too detailed
            
            Each question should be concise and clear, not exceeding 30 characters, directly related to expanding core knowledge.
            
            Please return only a list of questions, one per line.
            """
        else:
            # 中间层节点的提示词，侧重于连接核心和外围
            prompt = f"""
            You are analyzing a middle layer node in the knowledge gravitational field about "{self.topic}".
            
            Middle layer nodes connect the core and periphery, serving as bridges for knowledge transfer and transformation, between abstract principles and concrete applications.
            
            Please read the following information:
            {context}
            
            As '{role}', please generate {max_questions} follow-up questions that should:
            1. Connect abstract concepts with concrete applications
            2. Explore how principles are applied in practice
            3. Analyze connections or comparisons between different domains
            4. Balance the weight of theory and practice
            
            Each question should be concise and clear, not exceeding 30 characters, directly related to expanding middle layer knowledge.
            
            Please return only a list of questions, one per line.
            """
        
        try:
            response = self.llm.generate(prompt).strip()
            
            # 解析问题
            questions = [q.strip() for q in response.split('\n') if q.strip() and len(q) > 10]
            
            # 限制问题数量
            questions = questions[:max_questions]
            
            logger.info(f"从{level_info}节点生成了 {len(questions)} 个后续问题")
            return questions
            
        except Exception as e:
            logger.error(f"生成后续问题失败: {str(e)}")
            return []
        
    def _is_question_redundant(self, new_question, similarity_threshold=0.85):
        """检查新问题是否与现有问题重复"""
        try:
            # 计算新问题的嵌入
            new_embedding = self.encoder.encode([new_question])[0]
            
            # 与所有现有问题比较
            for node_id, node in self.nodes.items():
                existing_question = node.question_text
                
                # 如果节点ID在嵌入缓存中，使用缓存
                if node_id in self.embeddings:
                    existing_embedding = self.embeddings[node_id]
                else:
                    # 否则计算并缓存
                    existing_embedding = self.encoder.encode([existing_question])[0]
                    self.embeddings[node_id] = existing_embedding
                
                # 计算相似度
                similarity = cosine_similarity([new_embedding], [existing_embedding])[0][0]
                
                if similarity > similarity_threshold:
                    logger.info(f"问题冗余 (相似度: {similarity:.2f}): {existing_question[:30]}...")
                    return True
                    
            return False
        except Exception as e:
            logger.error(f"检查问题冗余性时出错: {str(e)}")
            # 出错时保守处理，不阻止问题添加
            return False

    def process_and_answer_new_questions(self, search_manager, answer_generator):
        """处理并回答引力场中的新问题"""
        # 找出所有没有答案的问题
        unanswered_nodes = {
            node_id: node for node_id, node in self.nodes.items()
            if not node.answer_text
        }
        
        if not unanswered_nodes:
            logger.info("没有需要回答的问题")
            return 0
            
        logger.info(f"处理 {len(unanswered_nodes)} 个未回答的问题")
        
        processed_count = 0
        for node_id, node in unanswered_nodes.items():
            try:
                # 搜索相关信息
                search_results = search_manager.search(node.question_text)
                
                # if not search_results:
                #     logger.warning(f"没有搜索结果: {node.question_text[:30]}...")
                #     continue
                    
                # 生成答案
                answer = answer_generator.generate_answer(
                    node.question_text, 
                    search_results, 
                    detailed=True
                )
                
                # 保存答案和搜索结果
                self.add_answer(node_id, answer)
                
                # 节点属性可能保存搜索结果，记录搜索结果，即使为空
                node.related_sources = search_results
                
                processed_count += 1
                # 判断答案是否来自直接生成
                if not search_results:
                    logger.info(f"直接生成问题答案(无搜索结果): {node.question_text[:30]}...")
                else:
                    logger.info(f"成功处理问题: {node.question_text[:30]}...")
                
            except Exception as e:
                logger.error(f"处理问题失败: {node.question_text[:30]}..., 错误: {str(e)}")
        
        if processed_count > 0:
            # 重建引力连接以反映新答案
            self.build_gravity_connections()
            
        logger.info(f"处理了 {processed_count}/{len(unanswered_nodes)} 个问题")
        return processed_count



















def process_questions_with_search(gravity_field, search_manager, answer_generator, concurrent=True, detailed=False):
    """Process all questions in the gravity field, search and generate answers
    
    Args:
        gravity_field: News gravity field instance
        search_manager: Search manager
        answer_generator: Answer generator
        concurrent: Whether to process concurrently
        detailed: Whether to generate detailed answers
        
    Returns:
        Number of successfully processed questions
    """
    # Get all questions without answers
    questions_to_process = [
        (node_id, node.question_text)
        for node_id, node in gravity_field.nodes.items()
        if not node.answer_text
    ]
    
    if not questions_to_process:
        logger.info("No questions to process")
        return 0
        
    logger.info(f"Starting to process {len(questions_to_process)} questions")
    

    #Define single question processing function
    def process_single_question(item):
        node_id, question = item
        try:
            # 1. Search
            search_results = search_manager.search(question)
            if not search_results:
                logger.warning(f"No search results for question: {question[:30]}...")
                return False
                
            # 2. Generate answer
            answer = answer_generator.generate_answer(question, search_results, detailed=detailed)
            
            # 3. Save answer and search results to node
            gravity_field.nodes[node_id].answer_text = answer
            gravity_field.nodes[node_id].related_sources = search_results
            
            # 4. Recalculate node mass
            gravity_field.calculate_node_mass(node_id)
            
            logger.info(f"Successfully processed question: {question[:30]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to process question: {question[:30]}..., error: {str(e)}")
            return False
    
    # Process all questions
    success_count = 0
    
    if concurrent and len(questions_to_process) > 1:
        # Concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(questions_to_process))) as executor:
            results = list(executor.map(process_single_question, questions_to_process))
            success_count = sum(1 for r in results if r)
    else:
        # Sequential processing
        for item in questions_to_process:
            if process_single_question(item):
                success_count += 1
    
    # Update gravity connections
    gravity_field.build_gravity_connections()
    
    logger.info(f"Question processing completed, successful: {success_count}/{len(questions_to_process)}")
    return success_count