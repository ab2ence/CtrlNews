import random
import threading
import time
import os
from openai import OpenAI
from typing import Optional, Literal, Any
from zhipuai import ZhipuAI
# This code is originally sourced from Repository STORM
# URL: [https://github.com/stanford-oval/storm]
import logging
class ChatGLM:
    """统一语言模型调用接口"""

    def __init__(self):
        self.API_KEY = ""  # 智谱AI平台获取
        self.MODEL_NAME = "glm-4-plus"  # 使用模型版本
        self.client = ZhipuAI(api_key=self.API_KEY)
        logging.info(f"ChatGLM模型初始化完成")

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                logging.info(f"ChatGLM模型生成完成")
                # 调试信息保留
                #print(f"\nresponse对象类型：{type(response)}")
                if hasattr(response, 'choices'):
                    print(f"choices数量：{len(response.choices)}")
                #print(f"choices内容：{response.choices}")
                # 修复后的解析逻辑
                if response.choices and len(response.choices) > 0:
                    first_choice = response.choices
                    #print(f"调用api返回的内容：{response.choices[0].message.content}")
                    return response.choices[0].message.content if hasattr(response.choices[0], 'message') else ""
                return None

            except Exception as e:
                print(f"API请求失败（尝试{attempt + 1}/{max_retries}）: {str(e)}")
        return None
    
class QwenMax:
    """统一语言模型调用接口"""

    def __init__(self):
        self.API_KEY = ""  # 智谱AI平台获取
        self.MODEL_NAME = "qwen-max-latest"  # 使用模型版本
        self.client = OpenAI(api_key=self.API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        logging.info(f"QwenMax模型初始化完成")

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                logging.info(f"QwenMax模型生成完成")
                # 调试信息保留
                #print(f"\nresponse对象类型：{type(response)}")
                if hasattr(response, 'choices'):
                    print(f"choices数量：{len(response.choices)}")
                #print(f"choices内容：{response.choices}")
                # 修复后的解析逻辑
                if response.choices and len(response.choices) > 0:
                    first_choice = response.choices
                    #print(f"调用api返回的内容：{response.choices[0].message.content}")
                    return response.choices[0].message.content if hasattr(response.choices[0], 'message') else ""
                return None

            except Exception as e:
                print(f"API请求失败（尝试{attempt + 1}/{max_retries}）: {str(e)}")
        return None

class DeepSeekV3:
    """Deepseek"""

    def __init__(self):
        self.MODEL_NAME = "deepseek-chat"  # 使用模型版本
        self.client = OpenAI(api_key="",
                              base_url="https://api.deepseek.com")
        logging.info(f"DeepSeekV3模型初始化完成")

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        logging.info(f"DeepSeekV3模型开始生成")
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )

                # 调试信息保留
                #print(f"\nresponse对象类型：{type(response)}")
                if hasattr(response, 'choices'):
                    print(f"choices数量：{len(response.choices)}")
                #print(f"choices内容：{response.choices}")
                # 修复后的解析逻辑
                if response.choices and len(response.choices) > 0:
                    first_choice = response.choices
                    #print(f"调用api返回的内容：{response.choices[0].message.content}")
                    return response.choices[0].message.content if hasattr(response.choices[0], 'message') else ""
                return None

            except Exception as e:
                print(f"API请求失败（尝试{attempt + 1}/{max_retries}）: {str(e)}")
        return None
    
class DeepSeekV3_Ali:
    """Deepseek"""

    def __init__(self):
        self.MODEL_NAME = "deepseek-v3"  # 使用模型版本
        self.client = OpenAI(api_key="",
                              base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        logging.info(f"DeepSeekV3_Ali模型初始化完成")

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        logging.info(f"DeepSeekV3_Ali模型开始生成")
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                logging.info(f"DeepSeekV3_Ali模型生成完成")
                time.sleep(0.5)
                # 调试信息保留
                #print(f"\nresponse对象类型：{type(response)}")
                if hasattr(response, 'choices'):
                    print(f"choices数量：{len(response.choices)}")
                #print(f"choices内容：{response.choices}")
                # 修复后的解析逻辑
                if response.choices and len(response.choices) > 0:
                    first_choice = response.choices
                    #print(f"调用api返回的内容：{response.choices[0].message.content}")
                    return response.choices[0].message.content if hasattr(response.choices[0], 'message') else ""
                return None

            except Exception as e:
                print(f"API请求失败（尝试{attempt + 1}/{max_retries}）: {str(e)}")
        return None

class DeepSeekR1_Ali:
    """Deepseek"""

    def __init__(self):
        self.MODEL_NAME = "deepseek-r1"  # 使用模型版本
        self.client = OpenAI(api_key="",
                              base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        logging.info(f"DeepSeekR1_Ali模型初始化完成")

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        logging.info(f"DeepSeekR1_Ali模型开始生成")
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                logging.info(f"DeepSeekR1_Ali模型生成完成")
                time.sleep(0.5)
                # 调试信息保留
                #print(f"\nresponse对象类型：{type(response)}")
                if hasattr(response, 'choices'):
                    print(f"choices数量：{len(response.choices)}")
                #print(f"choices内容：{response.choices}")
                # 修复后的解析逻辑
                if response.choices and len(response.choices) > 0:
                    first_choice = response.choices
                    #print(f"调用api返回的内容：{response.choices[0].message.content}")
                    return response.choices[0].message.content if hasattr(response.choices[0], 'message') else ""
                return None

            except Exception as e:
                print(f"API请求失败（尝试{attempt + 1}/{max_retries}）: {str(e)}")
        return None
    

class DeepSeekV3_Ali_v2:
    """Deepseek"""

    def __init__(self):
        self.MODEL_NAME = "deepseek-v3"  # 使用模型版本
        self.client = OpenAI(api_key="",
                              base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        logging.info(f"DeepSeekV3_Ali模型初始化完成")

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        logging.info(f"DeepSeekV3_Ali模型开始生成")
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                logging.info(f"DeepSeekV3_Ali模型生成完成")
                time.sleep(0.5)
                # 调试信息保留
                #print(f"\nresponse对象类型：{type(response)}")
                if hasattr(response, 'choices'):
                    print(f"choices数量：{len(response.choices)}")
                #print(f"choices内容：{response.choices}")
                # 修复后的解析逻辑
                if response.choices and len(response.choices) > 0:
                    first_choice = response.choices
                    #print(f"调用api返回的内容：{response.choices[0].message.content}")
                    return response.choices[0].message.content if hasattr(response.choices[0], 'message') else ""
                return None

            except Exception as e:
                print(f"API请求失败（尝试{attempt + 1}/{max_retries}）: {str(e)}")
        return None

class DeepSeekR1_Ali_v2:
    """Deepseek"""

    def __init__(self):
        self.MODEL_NAME = "deepseek-r1"  # 使用模型版本
        self.client = OpenAI(api_key="",
                              base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        logging.info(f"DeepSeekR1_Ali模型初始化完成")

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        logging.info(f"DeepSeekR1_Ali模型开始生成")
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                logging.info(f"DeepSeekR1_Ali模型生成完成")
                time.sleep(0.5)
                # 调试信息保留
                #print(f"\nresponse对象类型：{type(response)}")
                if hasattr(response, 'choices'):
                    print(f"choices数量：{len(response.choices)}")
                #print(f"choices内容：{response.choices}")
                # 修复后的解析逻辑
                if response.choices and len(response.choices) > 0:
                    first_choice = response.choices
                    #print(f"调用api返回的内容：{response.choices[0].message.content}")
                    return response.choices[0].message.content if hasattr(response.choices[0], 'message') else ""
                return None

            except Exception as e:
                print(f"API请求失败（尝试{attempt + 1}/{max_retries}）: {str(e)}")
        return None

class DeepSeekR1:
    """Deepseek"""

    def __init__(self):
        self.MODEL_NAME = "deepseek-reasoner"  # 使用模型版本
        self.client = OpenAI(api_key="",
                              base_url="https://api.deepseek.com")
        logging.info(f"DeepSeekR1模型初始化完成")

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        logging.info(f"DeepSeekR1模型开始生成")
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                logging.info(f"DeepSeekR1模型生成完成")
                # 调试信息保留
                #print(f"\nresponse对象类型：{type(response)}")
                if hasattr(response, 'choices'):
                    print(f"choices数量：{len(response.choices)}")
                #print(f"choices内容：{response.choices}")
                # 修复后的解析逻辑
                if response.choices and len(response.choices) > 0:
                    first_choice = response.choices
                    #print(f"调用api返回的内容：{response.choices[0].message.content}")
                    return response.choices[0].message.content if hasattr(response.choices[0], 'message') else ""
                return None

            except Exception as e:
                print(f"API请求失败（尝试{attempt + 1}/{max_retries}）: {str(e)}")
        return None
    

class GPT4o:
    """GPT4o"""

    def __init__(self):
        self.MODEL_NAME = "gpt-4o-2024-11-20"  # 使用模型版本
        self.client = OpenAI(api_key="",
                              base_url="https://api.openai-proxy.org/v1")
        logging.info(f"GPT4o模型初始化完成")

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                logging.info(f"GPT4o模型生成完成")

                # 调试信息保留
                #print(f"\nresponse对象类型：{type(response)}")
                if hasattr(response, 'choices'):
                    print(f"choices数量：{len(response.choices)}")
                #print(f"choices内容：{response.choices}")
                # 修复后的解析逻辑
                if response.choices and len(response.choices) > 0:
                    first_choice = response.choices
                    #print(f"调用api返回的内容：{response.choices[0].message.content}")
                    return response.choices[0].message.content if hasattr(response.choices[0], 'message') else ""
                return None

            except Exception as e:
                print(f"API请求失败（尝试{attempt + 1}/{max_retries}）: {str(e)}")
        return None
    
from anthropic import Anthropic

class Claude37Sonnet:
    """Claude37Sonnet"""

    def __init__(self):
        self.MODEL_NAME = "claude-3-7-sonnet-latest"  # 使用模型版本
        self.client = Anthropic(api_key="",
                              base_url="https://api.openai-proxy.org/anthropic")
        logging.info(f"Claude37Sonnet模型初始化完成")

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                logging.info(f"Claude37Sonnet模型生成完成")
                if hasattr(response, 'content'):
                    print(f"content数量：{len(response.content)}")
                
                # 提取纯文本内容
                if response.content and len(response.content) > 0:
                    # 收集所有文本块的内容
                    text_contents = []
                    for content_block in response.content:
                        if hasattr(content_block, 'text') and content_block.text:
                            text_contents.append(content_block.text)
                    
                    # 拼接所有文本
                    return '\n'.join(text_contents)
                
                return None

            except Exception as e:
                print(f"API请求失败（尝试{attempt + 1}/{max_retries}）: {str(e)}")
        return None
    

import google.generativeai as genai
from google.generativeai import types
class Gemini25:
    """Gemini25"""

    def __init__(self):
        genai.configure(
        api_key='',
        transport="rest",
        client_options={"api_endpoint": "https://api.openai-proxy.org/google"}
    )
        self.model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
        logging.info(f"Gemini25模型初始化完成")
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                )
                logging.info(f"Gemini25模型生成完成")
                # 通常response.text直接是字符串
                if hasattr(response, 'text'):
                    return response.text
                
                return None

            except Exception as e:
                print(f"API请求失败（尝试{attempt + 1}/{max_retries}）: {str(e)}")
        return None
