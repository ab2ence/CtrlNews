import sys
import os
import json
import time
import logging
from datetime import datetime
from tqdm import tqdm
import random
# 项目相关导入
from tools.duckduckgo_searchtool import DuckDuckGoSearch
from tools.lm import DeepSeekR1, DeepSeekV3, DeepSeekR1_Ali, DeepSeekV3_Ali
from simple_reviewer import SimpleArticleReviewer
from ArticleNewsWriter import ArticleNewsWriter
import random

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 导入引力场相关类
from GravitionalField import (
    NewsGravityField, 
    RoleCoordinator, 
    SearchManager, 
    AnswerGenerator, 
    process_questions_with_search
)
# 全局大模型
LLMHandler = DeepSeekV3_Ali()

class WorkflowNewsGenerator:
    """使用工作流生成新闻"""
    
    def __init__(self, topics_file="data/news_topics.json", output_dir="data/workflow"):
        """初始化新闻生成器
        
        参数:
            topics_file: 主题JSON文件路径
            output_dir: 输出目录
        """
        current_date = datetime.now().strftime("%Y%m%d_%H%M")
        self.topics_file = topics_file
        #self.output_dir = f"{output_dir}_{current_date}"
        self.output_dir = output_dir
        self.opinion_dir = os.path.join(self.output_dir, "opinion")  # 添加观点保存目录
        self.llm = LLMHandler
        self.retriever = DuckDuckGoSearch(k=3)
        self.simple_reviewer = SimpleArticleReviewer(DeepSeekR1_Ali())
        
        # 创建输出目录和观点目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.opinion_dir, exist_ok=True)  # 创建观点保存子目录
        
        # 加载主题
        self.topics = self._load_topics()
        logger.info(f"工作流新闻生成器初始化完成，加载了{len(self.topics)}个主题")
    
    def _load_topics(self):
        """从JSON文件加载主题"""
        try:
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("topics", [])
        except Exception as e:
            logger.error(f"加载主题文件失败: {str(e)}")
            return []
    
    def _generate_sentiment_control_params(self, terminal_nodes, sentiment_ratios):
        """根据情感比例为外围节点生成控制参数
        
        参数:
            terminal_nodes: 外围节点ID列表
            sentiment_ratios: 情感比例字典，如 {'positive': 0.33, 'neutral': 0.33, 'negative': 0.33'}
            
        返回:
            节点ID到参数配置的映射字典
        """
        # 确保比例总和为1
        total = sum(sentiment_ratios.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"情感比例总和不为1 ({total})，将进行归一化")
            sentiment_ratios = {k: v/total for k, v in sentiment_ratios.items()}
        
        # 定义情感参数映射
        viewpoint_params = {
            "positive": {"bias": 0.8, "emotion": 0.7, "exaggeration": 0.3},
            "neutral": {"bias": 0.0, "emotion": 0.0, "exaggeration": 0.1},
            "negative": {"bias": -0.8, "emotion": -0.7, "exaggeration": 0.7}
        }
        
        # 使用四舍五入而不是向下取整
        total_nodes = len(terminal_nodes)
        sentiment_counts = {}
        total_assigned = 0
        
        # 对所有非最后一个类型使用四舍五入
        sentiment_types = list(sentiment_ratios.keys())
        for i, (sentiment, ratio) in enumerate(sentiment_ratios.items()):
            if i == len(sentiment_types) - 1:
                # 最后一个类型分配剩余节点
                sentiment_counts[sentiment] = total_nodes - total_assigned
            else:
                # 其他类型使用四舍五入
                count = round(total_nodes * ratio)
                sentiment_counts[sentiment] = count
                total_assigned += count
        
        # 随机打乱节点顺序
        shuffled_nodes = list(terminal_nodes)
        random.shuffle(shuffled_nodes)
        
        # 分配情感参数
        result = {}
        node_index = 0
        
        for sentiment, count in sentiment_counts.items():
            params = viewpoint_params.get(sentiment, viewpoint_params["neutral"])
            for _ in range(count):
                if node_index < len(shuffled_nodes):
                    result[shuffled_nodes[node_index]] = params
                    node_index += 1
        
        return result

    def generate_news_for_topic(self, topic, topic_id, generations_per_topic=3, 
        sentiment_ratios={'positive': 0.3, 'neutral': 0.4, 'negative': 0.3},
        max_expansion_rounds=3, min_new_questions=2, 
        min_expansion_ratio=0.05, max_no_growth_count=1):
        """使用工作流为一个主题生成新闻，聚焦于末端节点的观点控制
        
        参数:
            topic: 主题字典，包含topic字段
            topic_id: 主题ID
            generations_per_topic: 每个主题生成多少篇文章
            sentiment_ratios: 情感比例字典，如 {'positive': 0.3, 'neutral': 0.4, 'negative': 0.3'}
            max_expansion_rounds: 最大知识扩展轮数
            min_new_questions: 最小新增问题数，低于此值终止循环
            min_expansion_ratio: 最小扩展比例，新增问题/现有问题低于此值终止循环
            max_no_growth_count: 连续扩展效果不增长的最大次数
        """
        topic_title = topic.get("topic", "Unknown Topic")
        
        logger.info(f"处理主题[{topic_id}]: {topic_title}")  
        logger.info(f"情感控制比例: 正面={sentiment_ratios.get('positive', 0):.2f}, 中性={sentiment_ratios.get('neutral', 0):.2f}, 负面={sentiment_ratios.get('negative', 0):.2f}")
        
        for gen_index in range(generations_per_topic):
            article_id = f"workflow_{topic_id}_{gen_index+1}"
            file_path = os.path.join(self.output_dir, f"{article_id}.txt")
            
            logger.info(f"生成文章: {article_id}")
            
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 1. 多智能体讨论，生成问题列表
                logger.info(f"[1/6] 正在进行多智能体讨论: {topic_title}")
                coordinator = RoleCoordinator(self.llm)
                questions, roles = coordinator.orchestrate_discussion(topic_title)
                logger.info(f"多智能体讨论生成了 {len(questions)} 个问题")

                # 2. 创建有向引力场并添加问题
                logger.info(f"[2/6] 正在构建有向引力场: {topic_title}")
                gravity_field = NewsGravityField(topic_title, llm=self.llm)
                
                # 添加问题到引力场
                for question, role in questions:
                    gravity_field.add_question(question, role)

                # 3. 搜索和回答问题
                logger.info(f"[3/6] 正在搜索和回答问题")
                search_manager = SearchManager(max_results=5)
                answer_generator = AnswerGenerator(self.llm)
                
                # 处理所有问题
                processed_count = gravity_field.process_and_answer_new_questions(
                    search_manager, 
                    answer_generator
                )
                logger.info(f"成功回答了 {processed_count} 个问题")
                
                # 4. 构建有向引力关系（但不生成观点）
                gravity_field.build_gravity_connections()
                
                logger.info(f"[4/6] 开始多轮知识图谱扩展")    
                # 初始化扩展统计数据
                total_expansion_rounds = 0
                total_new_questions = 0
                previous_new_questions = 0
                no_growth_count = 0
                # 多轮扩展循环
                for expansion_round in range(1, max_expansion_rounds + 1):
                    logger.info(f"执行第 {expansion_round}/{max_expansion_rounds} 轮知识扩展")
                    
                    # 获取当前节点数量作为基准
                    initial_node_count = len(gravity_field.nodes)
                    
                    # 5.1 扩展知识图谱，特别是外围节点
                    # 修改：prioritize_terminals 改为 prioritize_peripherals
                    new_questions = gravity_field.expand_knowledge(
                        max_new_questions=8, 
                        prioritize_peripherals=True
                    )
                    total_new_questions += new_questions
                    
                    logger.info(f"第 {expansion_round} 轮扩展产生了 {new_questions} 个新问题")
                    
                    # 如果没有生成新问题，增加未增长计数
                    if new_questions <= previous_new_questions:
                        no_growth_count += 1
                    else:
                        no_growth_count = 0
                    
                    previous_new_questions = new_questions
                    
                    # 检查是否应该终止循环
                    # 1. 新增问题数量太少
                    if new_questions < min_new_questions:
                        logger.info(f"新增问题数量 ({new_questions}) 低于阈值 ({min_new_questions})，终止扩展")
                        break
                        
                    # 2. 新增比例太低
                    expansion_ratio = new_questions / initial_node_count
                    if expansion_ratio < min_expansion_ratio:
                        logger.info(f"扩展比例 ({expansion_ratio:.3f}) 低于阈值 ({min_expansion_ratio})，终止扩展")
                        break
                        
                    # 3. 连续多次无增长
                    if no_growth_count > max_no_growth_count:
                        logger.info(f"连续 {no_growth_count} 轮扩展效果未增长，终止扩展")
                        break
                    
                    # 5.2 处理新生成的问题
                    if new_questions > 0:
                        new_processed = gravity_field.process_and_answer_new_questions(
                            search_manager, 
                            answer_generator
                        )
                        logger.info(f"成功回答了 {new_processed} 个新问题")
                        
                        # 重新构建引力关系
                        gravity_field.build_gravity_connections()
                    
                    total_expansion_rounds += 1
                    
                    # 如果已经达到最大轮数，退出循环
                    if expansion_round >= max_expansion_rounds:
                        logger.info(f"已达到最大扩展轮数 ({max_expansion_rounds})，终止扩展")
                        break
                
                logger.info(f"知识扩展完成，共执行 {total_expansion_rounds} 轮，生成 {total_new_questions} 个新问题")
                
                # 6. 获取所有外围节点（原末端节点）
                # 修改：使用 gravity_field.peripheral_nodes 替代 get_terminal_nodes()
                terminal_nodes = gravity_field.peripheral_nodes
                logger.info(f"图中共有 {len(terminal_nodes)} 个外围节点")
                
                # 7. 外围节点观点控制
                logger.info(f"[5/6] 正在为外围节点生成可控观点")
                
                # 7.1 根据情感比例生成外围节点的情感控制
                # 修改参数名：terminal_nodes 改为 peripheral_nodes
                terminal_control_params = self._generate_sentiment_control_params(
                    terminal_nodes, 
                    sentiment_ratios
                )
                    
                # 7.2 记录观点控制计划
                sentiment_stats = {'positive': 0, 'neutral': 0, 'negative': 0}
                terminal_control_info = {}
                
                for node_id, params in terminal_control_params.items():
                    node = gravity_field.nodes.get(node_id)
                    if node:
                        # 判断参数属于哪种情感类型
                        bias = params.get('bias', 0)
                        if bias > 0.3:
                            sentiment_type = 'positive'
                        elif bias < -0.3:
                            sentiment_type = 'negative'
                        else:
                            sentiment_type = 'neutral'
                            
                        terminal_control_info[node.question_text[:30]] = sentiment_type
                        sentiment_stats[sentiment_type] = sentiment_stats.get(sentiment_type, 0) + 1
                        
                logger.info(f"外围节点情感分布: 正面={sentiment_stats['positive']}, 中性={sentiment_stats['neutral']}, 负面={sentiment_stats['negative']}")

                # 7.3 为外围节点生成观点
                # 修改：使用 generate_peripheral_opinions 替代 generate_opinions_for_terminal_nodes
                generated_opinions = gravity_field.generate_peripheral_opinions(
                    sentiment_controls=terminal_control_params,
                    use_path_awareness=True
                )
                logger.info(f"成功为 {generated_opinions} 个外围节点生成观点")

                # 9. 使用ArticleNewsWriter生成文章
                logger.info(f"[6/6] 正在生成新闻文章")

                # 9.1 初始化ArticleNewsWriter
                # article_writer = ArticleNewsWriter(topic_title, self.llm)
                article_writer = ArticleNewsWriter(topic_title, DeepSeekR1_Ali())

                # 9.2 从引力场批量添加观点
                article_writer.add_opinions_from_gravity_field(gravity_field)
                
                # 9.3 生成文章
                default_params = {"length": 800}  # 目标字数
                result = article_writer.generate_article(default_params)

                # 保存用于生成文章的观点
                used_opinions_file_path = os.path.join(self.opinion_dir, f"{article_id}_used_opinions.json")
                used_opinions_data = {
                    "article_id": article_id,
                    "topic": topic_title,
                    "used_opinion_ids": result.get("used_opinion_ids", []),
                    "used_opinions_details": result.get("used_opinions_details", []),  # 新增：完整观点详情
                    "used_opinions_count": result.get("used_opinions", 0),
                    "total_opinions_count": result.get("total_opinions", 0)
                }

                # 保存使用的观点数据到JSON文件
                with open(used_opinions_file_path, 'w', encoding='utf-8') as f:
                    json.dump(used_opinions_data, f, ensure_ascii=False, indent=2)

                logger.info(f"已保存文章使用的 {result.get('used_opinions', 0)} 个观点到文件: {used_opinions_file_path}")
                
                # 10. 文章评审与润色
                original_content = result['content']
                
                # improved_content, _ = self.article_reviewer.iterative_article_improvement(
                #     original_content, 
                #     topic_title,
                #     max_iterations=3  # 3轮优化
                # )
                improved_content, _ = self.simple_reviewer.improve_article(
                    article=original_content, 
                    topic=topic_title, 
                    max_rounds=2,  # 只进行两轮优化
                    early_stop=True  # 启用提前终止
                )
                # 保存文章
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(improved_content)
                
                generation_time = time.time() - start_time
                logger.info(f"文章生成完成: {article_id}, 耗时: {generation_time:.2f}秒")
                
                # 添加短暂延迟
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"生成文章失败: {article_id}, 错误: {str(e)}")
                # 记录错误到文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"生成失败: {str(e)}")
    def generate_all_news(self, generations_per_topic=3,
                        sentiment_ratios={'positive': 0.3, 'neutral': 0.4, 'negative': 0.3},
                        max_expansion_rounds=3, 
                        min_new_questions=2, 
                        min_expansion_ratio=0.05, 
                        max_no_growth_count=1):
        
        """为选定的主题生成新闻"""
        if not self.topics:
            logger.error("没有加载任何主题，无法生成新闻")
            return
        selected_indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        start_time = datetime.now()
        logger.info(f"开始生成选定主题的新闻，时间: {start_time}")
        logger.info(f"将只处理主题索引: {selected_indices}")
        
        # 检查索引是否有效
        valid_indices = [idx for idx in selected_indices if 0 <= idx < len(self.topics)]
        if len(valid_indices) != len(selected_indices):
            invalid = set(selected_indices) - set(valid_indices)
            logger.warning(f"忽略无效的主题索引: {invalid}, 有效索引范围: 0-{len(self.topics)-1}")
        
        # 计算要生成的文章总数
        total_articles = len(valid_indices) * generations_per_topic
        logger.info(f"计划生成{total_articles}篇新闻文章")
        
        # 遍历所有主题
        with tqdm(total=total_articles, desc="总体进度") as pbar:
            for i, topic in enumerate(self.topics):
                # 只处理选定的主题索引
                if i not in valid_indices:
                    continue
                    
                topic_id = f"T{i+1}"
                logger.info(f"处理选定主题[{topic_id}]: {topic.get('topic', '未知')}")
                
                try:
                    self.generate_news_for_topic(
                        topic, 
                        topic_id, 
                        generations_per_topic,
                        sentiment_ratios,
                        max_expansion_rounds,
                        min_new_questions,
                        min_expansion_ratio,
                        max_no_growth_count
                    )
                    pbar.update(generations_per_topic)
                except Exception as e:
                    logger.error(f"处理主题失败: {topic.get('topic', '未知')}, 错误: {str(e)}")
                    pbar.update(generations_per_topic)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60.0
        logger.info(f"选定主题的新闻生成完成，耗时: {duration:.2f}分钟")

    
    # def generate_all_news(self, generations_per_topic=3,
    #                       max_expansion_rounds=3, 
    #                       min_new_questions=2, 
    #                       min_expansion_ratio=0.05, 
    #                       max_no_growth_count=1):
    #     """为所有主题生成新闻"""
    #     if not self.topics:
    #         logger.error("没有加载任何主题，无法生成新闻")
    #         return
        
    #     start_time = datetime.now()
    #     logger.info(f"开始生成所有新闻，时间: {start_time}")
        
    #     total_articles = len(self.topics) * generations_per_topic
    #     logger.info(f"计划生成{total_articles}篇新闻文章")
    #     # 遍历所有主题
    #     with tqdm(total=total_articles, desc="总体进度") as pbar:
    #         for i, topic in enumerate(self.topics):
    #             topic_id = f"T{i+1}"
    #             try:
    #                 self.generate_news_for_topic(topic, topic_id, 
    #                                              generations_per_topic,
    #                                              max_expansion_rounds, 
    #                                              min_new_questions, 
    #                                              min_expansion_ratio, 
    #                                              max_no_growth_count)
    #                 pbar.update(generations_per_topic)  # 每个主题生成generations_per_topic篇文章
    #             except Exception as e:
    #                 logger.error(f"处理主题失败: {topic.get('topic', '未知')}, 错误: {str(e)}")
    #                 pbar.update(generations_per_topic)
        
    #     end_time = datetime.now()
    #     duration = (end_time - start_time).total_seconds() / 60.0
    #     logger.info(f"所有新闻生成完成，耗时: {duration:.2f}分钟")


    def generate_all_news_index(self, generations_per_topic=3, start_topic_index=0,
                               sentiment_ratios={'positive': 0.3, 'neutral': 0.4, 'negative': 0.3}):
        """从指定索引开始为主题生成新闻"""
        if not self.topics:
            logger.error("没有加载任何主题，无法生成新闻")
            return
        
        # 检查起始索引是否有效
        if start_topic_index >= len(self.topics):
            logger.error(f"起始主题索引 {start_topic_index} 超出主题数量范围 (0-{len(self.topics)-1})")
            return
        
        # 计算有效的主题数量
        remaining_topics = len(self.topics) - start_topic_index
        total_articles = remaining_topics * generations_per_topic
        
        start_time = datetime.now()
        logger.info(f"开始从主题索引 {start_topic_index} (主题{start_topic_index+1})生成新闻，时间: {start_time}")
        logger.info(f"计划生成{total_articles}篇新闻文章")
        
        # 遍历所有主题，从指定索引开始
        with tqdm(total=total_articles, desc="总体进度") as pbar:
            for i in range(start_topic_index, len(self.topics)):
                topic = self.topics[i]
                topic_id = f"T{i+1}"  # 保持原有ID规则
                try:
                    self.generate_news_for_topic(
                        topic, 
                        topic_id, 
                        generations_per_topic,
                        sentiment_ratios
                    )
                    pbar.update(generations_per_topic)
                except Exception as e:
                    logger.error(f"处理主题失败: {topic.get('topic', '未知')}, 错误: {str(e)}")
                    pbar.update(generations_per_topic)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60.0
        logger.info(f"所有新闻生成完成，耗时: {duration:.2f}分钟")
# 主函数
def main():
    """主函数"""
    # 配置项
    topics_file = "./data/real_news_topics.json"
    output_dir = "./data/workflow_test_control"
    generations_per_topic = 2
    #start_topic_index = 19

    max_expansion_rounds = 1
    min_new_questions = 2
    min_expansion_ratio = 0.3
    max_no_growth_count = 1


    sentiment_ratios = {
        'positive': 0.33,  # 正面观点比例
        'neutral': 0.33,   # 中性观点比例
        'negative': 0.33   # 负面观点比例
    }

    
    # 初始化生成器
    generator = WorkflowNewsGenerator(topics_file, output_dir=output_dir)
    # 生成所有新闻
    generator.generate_all_news(generations_per_topic,
                                sentiment_ratios,
                                max_expansion_rounds, 
                                min_new_questions, 
                                min_expansion_ratio, 
                                max_no_growth_count)
    
    # 指定从哪个主题开始生成
   # generator.generate_all_news_index(generations_per_topic, start_topic_index)

if __name__ == "__main__":
    main()