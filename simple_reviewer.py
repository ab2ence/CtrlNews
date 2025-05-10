import logging
from typing import Dict, Tuple, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleArticleReviewer:
    """全面文章评审与润色工具 - 每轮覆盖全部优化维度"""
    
    def __init__(self, llm):
        """初始化文章评审工具
        
        参数:
            llm: 大语言模型实例，用于文章评审和润色
        """
        self.llm = llm
        # 定义三个核心关注领域
        self.focus_areas = [
            "Content relevance and structure",
            "Logical flow and coherence",
            "Expression clarity and readability"
        ]
    
    def improve_article(self, article: str, topic: str, max_rounds: int = 3, 
                        early_stop: bool = True) -> Tuple[str, Dict]:
        """多轮文章优化流程，每轮包含完整的评审和润色阶段
        
        参数:
            article: 原始文章内容
            topic: 文章主题
            max_rounds: 最大优化轮数
            early_stop: 是否在没有显著改进时提前终止
            
        返回:
            改进后的文章和处理元数据
        """
        logger.info(f"开始多轮全面文章优化: 主题'{topic}'")
        
        # 初始化变量
        current_article = article
        metadata = {
            "topic": topic,
            "original_length": len(article),
            "rounds": []
        }
        
        # 多轮优化循环
        for round_num in range(max_rounds):
            logger.info(f"开始第{round_num+1}轮全面优化")
            
            round_info = {
                "round": round_num + 1,
                "length_before": len(current_article),
                "focus_improvements": []
            }
            
            # 获取全面评审结果（同时覆盖所有关注点）
            comprehensive_review = self._comprehensive_review(current_article, topic)
            round_info["comprehensive_review"] = comprehensive_review
            
            # 执行全面润色
            improved_article = self._comprehensive_polish(current_article, topic, comprehensive_review)
            
            # 记录长度变化
            round_info["length_after"] = len(improved_article)
            round_info["change_percentage"] = round((len(improved_article) - len(current_article)) / len(current_article) * 100, 2)
            
            metadata["rounds"].append(round_info)
            
            # 检查是否有足够改进
            if early_stop and self._is_improvement_minimal(current_article, improved_article):
                logger.info(f"第{round_num+1}轮后未检测到显著改进，提前终止优化")
                round_info["stopped_early"] = True
                break
            
            # 更新当前文章版本
            current_article = improved_article
        
        # 完成优化
        metadata["final_length"] = len(current_article)
        metadata["total_rounds"] = len(metadata["rounds"])
        logger.info(f"文章优化完成: 共{metadata['total_rounds']}轮")
        
        return current_article, metadata
    
    def _comprehensive_review(self, article: str, topic: str) -> str:
        """对文章进行全面评审，基于评估的八个维度"""

        review_prompt = f"""
        As an expert article reviewer, analyze this article about "{topic}" based on these eight critical dimensions of quality.
        
        Article content:
        ```
        {article}
        ```
        
        Provide a detailed review covering:
        
        1. Overall assessment (3-4 sentences)
        
        2. Detailed analysis by dimension:
        
        A. Relevance
        - Identify sections where the article strays from the central topic
        - Note where content might be tangential or unnecessary
        - Suggest improvements to strengthen focus on the main theme
        
        B. Breadth
        - Identify significant aspects of the topic that are underdeveloped or missing
        - Note where additional perspectives would create more comprehensive coverage
        - Suggest areas where broader context could enhance understanding
        
        C. Depth
        - Highlight where analysis seems superficial or underdeveloped
        - Identify claims that need stronger supporting evidence or examples
        - Note opportunities to explore underlying mechanisms or relationships
        
        D. Novelty
        - Point out where conventional or clichéd approaches dominate
        - Identify opportunities to introduce fresh perspectives or insights
        - Suggest where innovative connections could be made
        
        E. Coherence
        - Identify paragraphs that don't connect smoothly with surrounding content
        - Note logical jumps where reader connections might break
        - Suggest improvements for narrative flow and structural organization
        
        F. Language Expression
        - Point out overly complex sentences that reduce clarity
        - Identify unclear expressions or ambiguous language
        - Note where more precise, engaging language would strengthen impact
        
        G. Topic Development
        - Assess how effectively the article develops its central theme
        - Identify weaknesses in the progression of ideas
        - Suggest improvements to the article's overall structure and development
        
        H. Intellectual Value
        - Note where the article could provide more insightful analysis
        - Identify opportunities to add thought-provoking ideas
        - Suggest connections to broader principles or implications
        
        3. Priority recommendations (in order of importance)
        - List 3-5 specific, actionable improvements that would most significantly enhance quality
        - Focus on changes that would address multiple dimensions simultaneously
        
        Be specific and constructive in your feedback, providing clear guidance for improvement.
        """

        return self.llm.generate(review_prompt)
    
    def _comprehensive_polish(self, article: str, topic: str, review: str) -> str:
        """基于评审意见对文章进行全面润色，改进所有八个维度"""

        polish_prompt = f"""
        As a skilled editor, improve this article about "{topic}" based on the provided expert review. Your goal is to create a piece that excels across all quality dimensions.
        
        Expert review:
        ```
        {review}
        ```
        
        Original article:
        ```
        {article}
        ```
        
        Apply these improvement principles in your revision:
        
        1. Relevance Enhancement:
        - Ensure every paragraph directly advances understanding of the central topic
        - Remove or revise content that doesn't contribute meaningfully to the main theme
        - Strengthen connections between specific points and the overall topic
        
        2. Breadth Improvement:
        - Address any significant aspects of the topic that were overlooked
        - Incorporate additional relevant perspectives where needed
        - Ensure balanced coverage of important dimensions of the topic
        
        3. Depth Development:
        - Strengthen analysis by adding supporting evidence where needed
        - Enhance explanation of underlying mechanisms and relationships
        - Add specific details or examples that illuminate complex points
        
        4. Novelty Enhancement:
        - Replace clichéd expressions with more original phrasing
        - Strengthen unique insights or perspectives
        - Create meaningful connections between ideas that offer fresh understanding
        
        5. Coherence Strengthening:
        - Add clear transitions between paragraphs and sections
        - Ensure logical progression throughout the article
        - Eliminate disruptions in flow or contradictions between sections
        
        6. Language Expression Refinement:
        - Simplify overly complex sentences without losing meaning
        - Replace vague language with precise, clear expressions
        - Vary sentence structure for improved rhythm and readability
        
        7. Topic Development Improvement:
        - Strengthen the introduction to better establish focus and significance
        - Ensure systematic progression of ideas throughout the article
        - Enhance the conclusion to effectively synthesize key insights
        
        8. Intellectual Value Addition:
        - Strengthen analytical components that encourage deeper thinking
        - Enhance connections to broader implications or principles
        - Refine expressions of insight to increase impact on readers
        
        Important guidelines:
        - Maintain the article's original scope and purpose
        - Don't add entirely new major points not found in the original
        - Preserve the overall word count (±5%)
        - Don't refer to the review or to the process of revision in your text
        
        Return only the revised article without any commentary.
        """

        return self.llm.generate(polish_prompt)
    
    def _is_improvement_minimal(self, original: str, improved: str) -> bool:
        """判断改进是否很小"""
        # 字数变化检查
        if abs(len(improved) - len(original)) < len(original) * 0.03:
            # 进一步使用LLM评估内容改进
            assess_prompt = f"""
            Please analyze these two versions of an article and determine if the second version shows significant, substantial improvements over the first.
            
            Original version (sample paragraphs):
            ```
            {self._get_sample_paragraphs(original)}
            ```
            
            Revised version (same sections):
            ```
            {self._get_sample_paragraphs(improved)}
            ```
            
            Consider the following aspects:
            1. Improvements in structure and organization
            2. Enhanced clarity of expression
            3. Strengthened logical coherence
            
            Please answer ONLY "Yes" or "No": Does the revised version show significant improvement?
            """
            
            result = self.llm.generate(assess_prompt).strip().lower()
            return "否" in result or "no" in result
        
        return False
    
    def _get_sample_paragraphs(self, text: str, max_chars: int = 800) -> str:
        """从文章中获取代表性段落样本"""
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if not paragraphs or len(text) <= max_chars:
            return text[:max_chars]
        
        # 选择开头、中间和结尾的段落
        samples = []
        if len(paragraphs) >= 1:
            samples.append(paragraphs[0])  # 开头
        if len(paragraphs) >= 3:
            samples.append(paragraphs[len(paragraphs) // 2])  # 中间
        if len(paragraphs) >= 2:
            samples.append(paragraphs[-1])  # 结尾
            
        return "\n\n".join(samples)[:max_chars]