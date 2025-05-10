import logging
import os
from typing import Callable, Union, List, Dict
import re
import uuid
import json
from utils.WebPageHelper import WebPageHelper

# 导入DuckDuckGo搜索库
from duckduckgo_search import DDGS

def clean_text(res):
    pattern = r'\[.*?\]\(.*?\)'
    result = re.sub(pattern, '', res)
    url_pattern = pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    result = re.sub(url_pattern, '', result)
    result = re.sub(r"\n\n+", "\n", result)
    return result

class DuckDuckGoSearch():
    def __init__(self, k=3, is_valid_source: Callable = None,
                 min_char_count: int = 150, snippet_chunk_size: int = 1000, 
                 webpage_helper_max_threads=10, region='wt-wt', 
                 safesearch='moderate', timelimit=None, **kwargs):
        """DuckDuckGo搜索引擎包装器
        
        参数:
            k: 每次搜索返回的最大结果数
            is_valid_source: 用于验证URL是否有效的函数
            min_char_count: 网页内容被认为有效的最小字符数
            snippet_chunk_size: 每个片段的最大字符数
            webpage_helper_max_threads: 网页内容获取的最大线程数
            region: 搜索地区，默认为全球(wt-wt)
            safesearch: 安全搜索级别，可选值: 'off', 'moderate', 'strict'
            timelimit: 搜索时间限制，如 'd' (天), 'w' (周), 'm' (月), 'y' (年)
            **kwargs: 其他DuckDuckGo搜索参数
        """
        # 初始化搜索引擎
        self.ddgs = DDGS()
        self.k = k
        self.region = region
        self.safesearch = safesearch
        self.timelimit = timelimit
        self.additional_params = kwargs
        
        # 初始化网页处理助手
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads
        )
        
        # 记录使用次数
        self.usage = 0
        
        # 设置URL验证函数
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True
    
    def get_usage_and_reset(self):
        """获取使用次数并重置"""
        usage = self.usage
        self.usage = 0
        return {'DuckDuckGoSearch': usage}
    
    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """使用DuckDuckGo搜索查询内容
        
        参数:
            query_or_queries: 单个查询字符串或查询列表
            exclude_urls: 需要排除的URL列表
            
        返回:
            结果列表，每个结果包含'description', 'snippets', 'title', 'url'等键
        """
        print("开始调用DuckDuckGo搜索引擎")
        
        # 确保查询是列表形式
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        
        # 增加使用计数
        self.usage += len(queries)
        
        # 存储URL对应的搜索结果
        url_to_results = {}
        
        # 对每个查询执行搜索
        for query in queries:
            try:
                # 使用DuckDuckGo搜索API执行查询
                results = self.ddgs.text(
                    query,
                    region=self.region,
                    safesearch=self.safesearch,
                    timelimit=self.timelimit,
                    max_results=self.k,
                    **self.additional_params
                )
                
                # 处理搜索结果
                for result in results:
                    url = result.get('href')
                    if url and self.is_valid_source(url) and url not in exclude_urls:
                        url_to_results[url] = {
                            'url': url,
                            'title': result.get('title', ''),
                            'description': result.get('body', '')
                        }
            except Exception as e:
                logging.error(f'DuckDuckGo搜索查询"{query}"时发生错误: {e}')
        
        # 使用WebPageHelper获取网页内容片段
        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(list(url_to_results.keys()))
        
        # 整合搜索结果和网页内容
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r['snippets'] = valid_url_to_snippets[url]['snippets']
            collected_results.append(r)
            
        return collected_results
    
    def __call__(self, query: Union[str, List[str]]) -> List[Dict]:
        """使对象可调用，直接调用forward方法"""
        return self.forward(query)