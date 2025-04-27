import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import requests
from requests.exceptions import RequestException
import math
import httpx
import logging
import re
from typing import Tuple, Dict, Any, List, Optional, Union
import tempfile
import time
import datetime

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 简单HTTP客户端类，用于在直接HTTP模式下提供兼容的客户端接口
class SimpleHTTPClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        # 创建一个兼容的chat对象，提供completions.create方法
        self.chat = SimpleCompletions(api_key, base_url)
        # 添加completions属性，与chat相同，确保兼容性
        self.completions = self.chat
    
    def get_models(self):
        """获取可用模型列表"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取模型列表失败: {str(e)}")
            return {"error": str(e)}

class SimpleCompletions:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        # 添加对自身的引用，确保completions.create也能正常工作
        self.completions = self
    
    def create(self, model="deepseek-chat", messages=None, temperature=0.7, max_tokens=2000, stream=False, response_format=None):
        """模拟OpenAI客户端的completions.create方法"""
        if not messages:
            messages = [{"role": "system", "content": "你是一个助手"}]
        
        # 确保消息格式正确
        validated_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            
            if "role" not in msg:
                continue
                
            if "content" not in msg:
                msg["content"] = ""
            
            # 确保content不是None
            if msg["content"] is None:
                msg["content"] = ""
                
            # 确保role是有效值
            if msg["role"] not in ["system", "user", "assistant"]:
                msg["role"] = "user"  # 默认为用户角色
            
            # 转换为字符串
            content = msg["content"]
            if not isinstance(content, str):
                content = str(content)
                
            # 添加清理后的消息
            validated_messages.append({
                "role": msg["role"],
                "content": content
            })
        
        # 如果没有有效的消息，添加一个默认消息
        if not validated_messages:
            validated_messages = [{"role": "user", "content": "请提供一个友好的回应"}]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,
            "messages": validated_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # 添加response_format参数支持
        if response_format:
            payload["response_format"] = response_format
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                stream=stream,
                timeout=60
            )
            
            # 检查错误状态码
            if response.status_code != 200:
                error_detail = ""
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        error_detail = error_json["error"].get("message", "")
                except:
                    error_detail = response.text[:100]  # 获取部分响应内容
                
                raise Exception(f"API错误: {response.status_code} - {error_detail}")
            
            if stream:
                # 返回一个生成器，模拟流式输出
                def generate_stream():
                    for line in response.iter_lines():
                        if line:
                            try:
                                if line.strip() == b"data: [DONE]":
                                    break
                                if line.startswith(b"data: "):
                                    json_data = json.loads(line[6:])
                                    # 创建一个类似于OpenAI客户端返回的对象
                                    yield SimpleResponse(json_data)
                            except Exception as e:
                                print(f"解析流失败: {str(e)}")
                return generate_stream()
            else:
                # 非流式返回
                result = response.json()
                return SimpleResponse(result)
        except requests.RequestException as e:
            raise Exception(f"网络错误: {str(e)}")
        except json.JSONDecodeError:
            raise Exception(f"API返回了无效的JSON响应: {response.text[:100]}...")
        except Exception as e:
            raise Exception(f"API请求失败: {str(e)}")

class SimpleResponse:
    """模拟OpenAI客户端响应对象"""
    def __init__(self, data):
        self.data = data
        if "choices" in data:
            self.choices = [SimpleChoice(choice) for choice in data["choices"]]
        else:
            self.choices = []

class SimpleChoice:
    """模拟OpenAI客户端Choice对象"""
    def __init__(self, data):
        self.data = data
        if "message" in data:
            self.message = SimpleMessage(data["message"])
        elif "delta" in data:
            self.delta = SimpleMessage(data["delta"])

class SimpleMessage:
    """模拟OpenAI客户端Message对象"""
    def __init__(self, data):
        self.data = data
        # 确保content存在并为字符串
        if "content" in data:
            content = data.get("content", "")
            self.content = str(content) if content is not None else ""
        else:
            self.content = ""
        
        # 支持其他可能的字段(但排除reasoning_content)
        for key, value in data.items():
            if key not in ["content"]:
                setattr(self, key, value)

class DeepSeekAPI:
    def __init__(self, api_key: str = None, api_url: str = None):
        """初始化DeepSeek API客户端
        
        Args:
            api_key: DeepSeek API密钥，默认从环境变量读取
            api_url: DeepSeek API URL，默认为官方API地址
        """
        # 如果未提供API密钥，则从环境变量获取
        self.api_key = api_key if api_key else os.getenv("DEEPSEEK_API_KEY", "")
        self.api_url = api_url if api_url else os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        
        if not self.api_key:
            raise ValueError("API密钥未提供，请设置DEEPSEEK_API_KEY环境变量或直接传入api_key参数")
        
        # 设置代理（如果需要）
        http_proxy = os.getenv("HTTP_PROXY", "")
        https_proxy = os.getenv("HTTPS_PROXY", "")
        
        # 创建OpenAI客户端
        try:
            # 创建标准OpenAI客户端，不使用代理
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_url
            )
            
            # 如果有代理设置，改用httpx客户端
            if http_proxy or https_proxy:
                try:
                    proxies = {}
                    if http_proxy:
                        proxies["http://"] = http_proxy
                    if https_proxy:
                        proxies["https://"] = https_proxy
                    
                    # 创建httpx客户端
                    http_client = httpx.Client(proxies=proxies)
                    
                    # 使用http_client创建OpenAI客户端
                    # 注意：OpenAI客户端接受http_client参数，不是proxies
                    self.client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.api_url,
                        http_client=http_client
                    )
                    logger.info("使用代理设置创建了OpenAI客户端")
                except Exception as proxy_error:
                    logger.warning(f"设置代理时出错: {str(proxy_error)}，使用无代理客户端")
                    # 出错时回退到无代理客户端
                    self.client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.api_url
                    )
            
            logger.info(f"成功创建DeepSeek API客户端，API URL: {self.api_url}")
        except Exception as e:
            logger.error(f"创建DeepSeek API客户端时出错: {str(e)}")
            # 创建一个简单HTTP客户端作为备选
            try:
                self.client = SimpleHTTPClient(self.api_key, self.api_url)
                logger.info("回退至简单HTTP客户端")
            except Exception as fallback_error:
                logger.error(f"创建备选客户端时出错: {str(fallback_error)}")
                raise e
    
    def list_models(self):
        """获取可用模型列表
        
        Returns:
            模型列表响应对象
        """
        try:
            # 使用OpenAI客户端获取模型列表
            response = self.client.models.list()
            logger.info("成功获取模型列表")
            return response
        except Exception as e:
            logger.error(f"获取模型列表时出错: {str(e)}")
            # 尝试使用简单HTTP客户端作为回退方法
            try:
                if hasattr(self.client, "get_models"):
                    return self.client.get_models()
                else:
                    # 手动发送HTTP请求
                    response = requests.get(
                        f"{self.api_url}/models",
                        headers={"Authorization": f"Bearer {self.api_key}"}
                    )
                    response.raise_for_status()
                    return response.json()
            except Exception as fallback_error:
                logger.error(f"获取模型列表的回退方法也失败: {str(fallback_error)}")
                return {"error": str(e)}
    
    def generate_response(self, prompt: str, model: str = "deepseek-chat") -> Tuple[str, str]:
        """调用DeepSeek API获取回答
        
        Args:
            prompt: 提示文本
            model: 模型名称，默认为"deepseek-chat"（仅支持此模型）
            
        Returns:
            包含回答内容和空字符串的元组(answer, "")
        """
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": "你是一个专业的数据处理和分析助手。"},
                {"role": "user", "content": prompt}
            ]
            
            # 固定使用deepseek-chat模型
            use_model = "deepseek-chat"
            
            # 调用API（非流式）
            response = self.client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
            
            # 获取回答内容
            content = response.choices[0].message.content
            
            # 返回回答内容与空字符串
            return content, ""
        except Exception as e:
            logger.error(f"调用DeepSeek API时出错: {str(e)}")
            return f"API调用错误: {str(e)}", ""
            
    def stream_chat_completion(self, messages: List[Dict[str, str]], model: str = "deepseek-chat") -> Tuple[str, str]:
        """流式调用DeepSeek API获取回答
        
        Args:
            messages: 消息历史列表
            model: 模型名称，默认为"deepseek-chat"（目前仅支持此模型）
            
        Returns:
            包含答案内容和空字符串的元组(answer, "")
        """
        # 确保消息中的所有内容都可以被JSON序列化
        processed_messages = []
        
        # 辅助函数：处理任何类型的值，确保其可以被JSON序列化
        def process_value(val):
            # 处理None值
            if val is None:
                return ""
            # 处理时间戳类型
            if isinstance(val, (pd.Timestamp, np.datetime64)):
                return val.isoformat() if hasattr(val, 'isoformat') else str(val)
            # 处理numpy数值类型
            elif isinstance(val, np.integer):
                return int(val)
            elif isinstance(val, np.floating):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            # 处理字典
            elif isinstance(val, dict):
                return {k: process_value(v) for k, v in val.items()}
            # 处理列表或元组
            elif isinstance(val, (list, tuple)):
                return [process_value(item) for item in val]
            # 处理其他非基本类型
            elif not isinstance(val, (str, int, float, bool, type(None))):
                return str(val)
            else:
                return val
        
        # 处理消息列表
        for msg in messages:
            if not isinstance(msg, dict):
                logger.warning(f"跳过非字典消息: {type(msg)}")
                continue
            
            # 检查必要的role和content字段
            if "role" not in msg:
                logger.warning(f"消息缺少'role'字段: {msg}")
                continue
                
            if "content" not in msg:
                logger.warning(f"消息缺少'content'字段: {msg}")
                # 对于没有content的消息，添加空字符串
                msg["content"] = ""
            
            # 确保content不是None
            if msg["content"] is None:
                msg["content"] = ""
                
            # 确保role是有效值
            if msg["role"] not in ["system", "user", "assistant"]:
                logger.warning(f"消息角色无效: {msg['role']}")
                msg["role"] = "user"  # 默认为用户角色
            
            processed_msg = {}
            # 仅保留必要字段：role和content
            processed_msg["role"] = process_value(msg["role"])
            processed_msg["content"] = process_value(msg["content"])
            
            processed_messages.append(processed_msg)
        
        # 如果没有有效的消息，添加一个默认的用户消息
        if not processed_messages:
            processed_messages = [{"role": "user", "content": "请提供一个友好的回应"}]
        
        try:
            # 固定使用deepseek-chat模型
            use_model = "deepseek-chat"
                
            logger.info(f"尝试流式调用API，使用模型: {use_model}")
            
            # 尝试流式API调用
            try:
                logger.info(f"发送API请求: 模型={use_model}, 消息数量={len(processed_messages)}")
                # 记录消息结构以便调试
                for i, msg in enumerate(processed_messages):
                    logger.info(f"消息 {i}: role={msg.get('role', 'unknown')}, content长度={len(str(msg.get('content', '')))}")
                
                # 使用选择的模型创建流式请求
                stream = self.client.chat.completions.create(
                    model=use_model,  # 使用deepseek-chat模型
                    messages=processed_messages,
                    temperature=0.7,
                    max_tokens=2000,
                    stream=True
                )
                
                # 收集回答内容
                content = ""
                
                # 处理流式输出
                for chunk in stream:
                    # 检查是否有回答内容
                    if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
                        content += chunk.choices[0].delta.content
                
                return content, ""
            except Exception as stream_error:
                logger.error(f"流式API调用失败: {str(stream_error)}，尝试非流式调用")
                # 不抛出异常，而是继续执行后续回退逻辑
        
        except Exception as e:
            logger.error(f"流式调用DeepSeek API时出错: {str(e)}")
            
        # 尝试使用非流式回退方法
        try:
            logger.info("尝试使用非流式API作为回退方法")
            
            # 如果消息太长可能导致错误，尝试减少消息数量并限制内容长度
            if len(processed_messages) > 5:
                logger.info(f"消息太多，截取最近的5条消息")
                # 确保保留系统消息
                system_messages = [msg for msg in processed_messages if msg.get("role") == "system"]
                # 最多保留一条系统消息
                if len(system_messages) > 1:
                    system_messages = [system_messages[0]]
                
                # 取最后几条非系统消息
                non_system_messages = [msg for msg in processed_messages if msg.get("role") != "system"]
                recent_messages = non_system_messages[-4:] if len(non_system_messages) > 4 else non_system_messages
                
                # 组合消息并限制内容长度
                processed_messages = []
                for msg in system_messages + recent_messages:
                    # 限制消息内容长度
                    if isinstance(msg.get("content"), str) and len(msg["content"]) > 4000:
                        msg["content"] = msg["content"][:4000] + "..."
                    processed_messages.append(msg)
            
            logger.info(f"发送非流式请求: 模型=deepseek-chat, 消息数量={len(processed_messages)}")
            
            try:
                # 非流式API调用
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=processed_messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # 获取内容
                content = response.choices[0].message.content
                
                return content, ""
            except Exception as e:
                logger.error(f"非流式API调用失败: {str(e)}")
                # 继续执行到最后的回退方法
        except Exception as fallback_error:
            logger.error(f"非流式回退方法失败: {str(fallback_error)}")
        
        # 最后的回退: 简化HTTP请求
        try:
            # 尝试使用最简单的HTTP请求作为最终回退
            logger.info("尝试使用最简单的HTTP请求作为最终回退")
            
            # 创建极简消息
            simple_messages = []
            
            # 尝试找到一条系统消息
            system_msg = next((msg for msg in processed_messages if msg.get("role") == "system"), None)
            if system_msg:
                simple_content = str(system_msg.get("content", ""))
                # 限制长度
                if len(simple_content) > 500:
                    simple_content = simple_content[:500]
                simple_messages.append({"role": "system", "content": simple_content})
            
            # 尝试找到最后一条用户消息
            user_msgs = [msg for msg in processed_messages if msg.get("role") == "user"]
            if user_msgs:
                simple_content = str(user_msgs[-1].get("content", ""))
                # 限制长度
                if len(simple_content) > 500:
                    simple_content = simple_content[:500]
                simple_messages.append({"role": "user", "content": simple_content})
            
            # 如果没有找到任何消息，使用默认消息
            if not simple_messages:
                simple_messages = [{"role": "user", "content": "请提供一个友好的回应"}]
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 使用最稳定的模型
            payload = {
                "model": "deepseek-chat",
                "messages": simple_messages,
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            logger.info(f"发送直接HTTP请求，消息数量: {len(simple_messages)}")
            
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            content = result["choices"][0]["message"]["content"]
            return content, ""
        except Exception as final_error:
            logger.error(f"最终回退方法也失败: {str(final_error)}")
            # 完全无法获取响应，返回错误消息
            return "抱歉，无法处理您的请求。请检查API设置或稍后再试。", ""

    def generate_table_json(self, input_data: Dict, prompt: str) -> Dict:
        """生成表格数据的JSON输出
        
        Args:
            input_data: 输入数据字典，包含表格相关信息
            prompt: 用户提示，说明想要生成什么样的表格
            
        Returns:
            解析后的JSON对象，包含表格数据
        """
        try:
            # 构建消息
            system_prompt = """你是一个专业的数据分析助手，精通将数据转换为结构化表格。

请根据用户提供的数据信息和要求，以JSON格式输出标准的表格数据。你的输出必须严格遵循以下格式：

{
    "table_name": "表格名称（根据用户需求和数据内容确定）",
    "headers": ["列1", "列2", "列3", ...],
    "data": [
        ["行1列1值", "行1列2值", "行1列3值", ...],
        ["行2列1值", "行2列2值", "行2列3值", ...],
        ...
    ],
    "summary": "表格数据的简明摘要，描述表格内容和目的"
}

请注意：
1. 表格应该根据用户的需求进行设计，反映数据的关键特点
2. 确保数据类型合理，数值型数据应保持为数值，不要转为字符串
3. 仅输出JSON格式数据，不要添加任何额外的解释或说明文字
4. 数据处理应当准确，如有需要可进行合理的计算和汇总"""
            
            # 准备更结构化的用户提示
            user_content = f"""请根据以下数据信息创建一个表格：

## 数据概况
- 总行数: {input_data.get('row_count', 0)}
- 可用列名: {', '.join(input_data.get('columns', []))}
- 数值型列: {', '.join(input_data.get('numeric_columns', []))}

## 数据样本
{json.dumps(input_data.get('sample_data', []), ensure_ascii=False, indent=2)}

## 用户要求
{prompt}

请严格按照JSON格式输出表格数据，不要添加任何额外的解释。"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            # 检查是否使用SimpleHTTPClient
            is_simple_client = isinstance(self.client, SimpleHTTPClient)
            
            try:
                # 调用API指定JSON输出格式
                if is_simple_client:
                    # 对于SimpleHTTPClient，我们在消息中强调JSON格式，但不使用response_format
                    messages[0]["content"] += "\n\n重要：你必须只输出JSON格式，不要有任何其他文本。不要包含解释或代码块，只有纯JSON。"
                    messages[1]["content"] += "\n\n请确保你的回复只包含纯JSON，不要有其他任何文本（如'```json'或解释）。"
                    
                    response = self.client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=4000
                    )
                else:
                    # 对于标准OpenAI客户端，使用response_format
                    response = self.client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        response_format={'type': 'json_object'},
                        max_tokens=4000
                    )
                
                # 解析JSON响应
                content = response.choices[0].message.content
                
                # 处理可能的JSON格式问题
                content = content.strip()
                
                # 移除可能的Markdown代码块标记
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                
                # 移除空白行和内容开头的可能干扰字符
                content = content.strip()
                
                logger.info(f"成功生成表格JSON数据")
                return json.loads(content)
            
            except Exception as api_error:
                logger.error(f"首次生成表格JSON数据失败: {str(api_error)}")
                
                # 第二次尝试：使用更简单的提示和普通文本响应
                try:
                    # 简化提示，更强调格式要求
                    simple_system_prompt = "你是一个JSON格式化工具。只输出纯JSON，没有任何额外的文本或解释。"
                    simple_user_prompt = f"""根据以下数据和要求生成一个JSON格式的表格：
                    
                    数据：
                    - 总行数: {input_data.get('row_count', 0)}
                    - 列: {', '.join(input_data.get('columns', []))}
                    
                    需求: {prompt}
                    
                    只输出一个有效的JSON对象，格式为:
                    {{
                        "table_name": "表名",
                        "headers": ["列1", "列2"],
                        "data": [["值1", "值2"], ["值3", "值4"]],
                        "summary": "描述"
                    }}
                    
                    不要包含```或其他标记，只输出纯JSON。"""
                    
                    simple_messages = [
                        {"role": "system", "content": simple_system_prompt},
                        {"role": "user", "content": simple_user_prompt}
                    ]
                    
                    # 尝试直接使用请求库
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    
                    payload = {
                        "model": "deepseek-chat",
                        "messages": simple_messages,
                        "temperature": 0.5,
                        "max_tokens": 2000
                    }
                    
                    response = requests.post(
                        f"{self.api_url}/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # 清理内容，确保它是一个有效的JSON
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    
                    return json.loads(content)
                    
                except Exception as fallback_error:
                    logger.error(f"第二次尝试也失败: {str(fallback_error)}")
                    
                    # 最后回退：生成一个基本的表格结构
                    try:
                        # 创建一个基本的表格结构
                        headers = input_data.get('columns', [])[:5]  # 最多取5列
                        
                        if not headers:
                            headers = ["项目", "值"]
                        
                        # 创建一些基本数据
                        data = []
                        for i in range(min(3, len(input_data.get('sample_data', [])))):
                            row = []
                            for h in headers:
                                if h in input_data.get('sample_data', [])[i]:
                                    row.append(input_data.get('sample_data', [])[i][h])
                                else:
                                    row.append("")
                            data.append(row)
                        
                        # 如果没有样本数据，创建一些占位符
                        if not data:
                            data = [[""] * len(headers) for _ in range(3)]
                        
                        # 创建基本表格
                        basic_table = {
                            "table_name": "基本数据表",
                            "headers": headers,
                            "data": data,
                            "summary": "自动生成的基本数据表，由于API限制无法生成完整表格"
                        }
                        
                        return basic_table
                    
                    except:
                        # 最终回退：返回错误信息
                        return {
                            "error": str(api_error),
                            "table_name": "错误",
                            "headers": ["错误信息"],
                            "data": [[str(api_error)]],
                            "summary": "生成表格时出错"
                        }
        except Exception as e:
            logger.error(f"生成表格JSON数据时出错: {str(e)}")
            return {"error": str(e)}

class DormDataProcessor:
    def __init__(self, api_key: str = None, api_url: str = None):
        """初始化宿舍数据处理器
        
        Args:
            api_key: DeepSeek API密钥，默认从环境变量读取
            api_url: DeepSeek API URL，默认为官方API地址
        """
        # 初始化DeepSeek API
        try:
            self.api = DeepSeekAPI(api_key, api_url)
            logger.info("成功初始化DormDataProcessor")
        except Exception as e:
            logger.error(f"初始化DormDataProcessor时出错: {str(e)}")
            self.api = None
            
    def process_excel(self, file_path: str, calculate: bool = True) -> pd.DataFrame:
        """处理Excel文件
        
        Args:
            file_path: Excel文件路径
            calculate: 是否计算费用，默认为True
            
        Returns:
            处理后的DataFrame
        """
        try:
            # 读取Excel文件
            df = pd.read_excel(file_path)
            logger.info(f"成功读取Excel文件，共{len(df)}行数据")
            
            # 如果需要计算费用
            if calculate:
                df = self.calculate_fees(df)
                logger.info("完成费用计算")
                
            return df
        except Exception as e:
            logger.error(f"处理Excel文件时出错: {str(e)}")
            raise
    
    def calculate_fees(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算费用
        
        Args:
            df: 输入的DataFrame
            
        Returns:
            计算后的DataFrame
        """
        try:
            # 自动检测列名并标准化
            self._standardize_column_names(df)
            
            # 验证必要的列是否存在
            required_columns = ['入住天数', '合计天数', '宿舍水电费']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"缺少必要的列: {col}")
            
            # 确保数据类型正确
            for col in ['入住天数', '合计天数', '宿舍水电费']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 处理租金列（如果存在）
            if '个人租金' in df.columns:
                df['个人租金'] = pd.to_numeric(df['个人租金'], errors='coerce')
                df['个人租金'] = df['个人租金'].fillna(0)
            else:
                df['个人租金'] = 0
            
            # 计算个人水电费
            df['个人水电费'] = (df['入住天数'] / df['合计天数'] * df['宿舍水电费']).round(2)
            
            # 计算合计费用
            df['合计'] = (df['个人水电费'] + df['个人租金']).round(2)
            
            return df
        except Exception as e:
            logger.error(f"计算费用时出错: {str(e)}")
            raise
    
    def calculate_fees_using_api(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """使用API计算费用，并提供更详细的自费人员费用计算
        
        Args:
            df: 输入的DataFrame
            
        Returns:
            计算后的DataFrame和处理消息
        """
        try:
            # 复制DataFrame以避免修改原始数据
            result_df = df.copy()
            
            # 自动检测列名并标准化
            self._standardize_column_names(result_df)
            
            # 验证必要的列是否存在
            required_columns = ['入住天数', '合计天数', '宿舍水电费']
            for col in required_columns:
                if col not in result_df.columns:
                    return result_df, f"缺少必要的列: {col}，无法计算费用"
            
            # 确保数据类型正确
            for col in ['入住天数', '合计天数', '宿舍水电费']:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            
            # 检查是否有'入住性质'列或相关标识，以识别自费人员
            is_self_paying = False
            has_rent_column = False
            
            if '入住性质' in result_df.columns:
                # 查找自费人员
                self_paying_rows = result_df['入住性质'].astype(str).str.contains('自费')
                if self_paying_rows.any():
                    is_self_paying = True
                    logger.info(f"发现{self_paying_rows.sum()}名自费住宿人员")
            
            # 检查是否有租金相关列
            rent_columns = [col for col in result_df.columns if '租金' in col and col != '个人租金']
            if rent_columns:
                has_rent_column = True
                # 选择第一个租金列作为宿舍租金
                rent_column = rent_columns[0]
                result_df[rent_column] = pd.to_numeric(result_df[rent_column], errors='coerce')
                logger.info(f"发现租金列: {rent_column}")
            
            # 处理床位数列
            if '床位数' in result_df.columns:
                result_df['床位数'] = pd.to_numeric(result_df['床位数'], errors='coerce')
                # 确保床位数至少为1
                result_df['床位数'] = result_df['床位数'].fillna(1)
                result_df.loc[result_df['床位数'] <= 0, '床位数'] = 1
            elif is_self_paying:
                # 如果有自费人员但没有床位数列，添加默认床位数
                result_df['床位数'] = 1
                logger.info("未找到床位数列，默认设置为1")
            
            # 计算个人水电费
            result_df['个人水电费'] = (result_df['入住天数'] / result_df['合计天数'] * result_df['宿舍水电费']).round(2)
            
            # 对自费人员计算个人租金
            if is_self_paying and has_rent_column:
                # 只为标记为自费的人员计算租金
                self_paying_mask = result_df['入住性质'].astype(str).str.contains('自费')
                
                # 计算个人租金 = 租金 ÷ 床位数 × 入住天数 ÷ 合计天数
                result_df.loc[self_paying_mask, '个人租金'] = (
                    result_df.loc[self_paying_mask, rent_column] / 
                    result_df.loc[self_paying_mask, '床位数'] * 
                    result_df.loc[self_paying_mask, '入住天数'] / 
                    result_df.loc[self_paying_mask, '合计天数']
                ).round(2)
                
                logger.info("完成自费人员个人租金计算")
            elif '个人租金' in result_df.columns:
                # 如果已有个人租金列，确保为数值类型
                result_df['个人租金'] = pd.to_numeric(result_df['个人租金'], errors='coerce')
                result_df['个人租金'] = result_df['个人租金'].fillna(0)
            else:
                # 否则添加个人租金列并设为0
                result_df['个人租金'] = 0
            
            # 计算合计费用
            result_df['合计'] = (result_df['个人水电费'] + result_df['个人租金']).round(2)
            
            # 生成处理消息
            message = f"成功计算{len(result_df)}条记录的费用。"
            if is_self_paying:
                message += f" 其中包含自费住宿人员。"
            
            return result_df, message
        except Exception as e:
            logger.error(f"使用API计算费用时出错: {str(e)}")
            return df, f"计算费用时出错: {str(e)}"
    
    def fill_missing_data(self, df: pd.DataFrame, model: str = "deepseek-chat") -> pd.DataFrame:
        """填补缺失数据
        
        Args:
            df: 输入的DataFrame
            model: 模型名称，默认为"deepseek-chat"
            
        Returns:
            填补后的DataFrame
        """
        try:
            # 复制DataFrame以避免修改原始数据
            result_df = df.copy()
            
            # 使用AI填补缺失数据
            if self.api:
                # 自动检测缺失严重的列
                missing_columns = result_df.columns[result_df.isnull().mean() > 0.1]
                
                if len(missing_columns) > 0:
                    logger.info(f"发现{len(missing_columns)}列缺失数据较多: {', '.join(missing_columns)}")
                    
                    # 对缺失数据较多的列使用AI预测
                    for col in missing_columns:
                        # 跳过不适合预测的列
                        if col in ['序号', '合计', '个人水电费', '个人租金']:
                            continue
                        
                        # 获取有缺失值的行
                        missing_rows = result_df[result_df[col].isnull()]
                        
                        if len(missing_rows) > 0:
                            logger.info(f"尝试填补'{col}'列的{len(missing_rows)}个缺失值")
                            
                            # 对每个缺失值行使用AI进行预测
                            for idx, row in missing_rows.iterrows():
                                # 构建提示
                                prompt = f"""基于以下宿舍数据行，预测缺失的'{col}'值：
                                {row.to_dict()}
                                
                                请只返回预测值，不要解释。如果无法预测，返回'NA'。"""
                                
                                # 调用API获取预测值
                                predicted_value, _ = self.api.generate_response(prompt, model)
                                
                                # 清理预测值
                                predicted_value = predicted_value.strip()
                                if predicted_value.lower() == 'na':
                                    continue
                                
                                # 尝试将预测值转换为适当的类型
                                if result_df[col].dtype == 'int64':
                                    try:
                                        result_df.at[idx, col] = int(predicted_value)
                                    except:
                                        pass
                                elif result_df[col].dtype == 'float64':
                                    try:
                                        result_df.at[idx, col] = float(predicted_value)
                                    except:
                                        pass
                                else:
                                    result_df.at[idx, col] = predicted_value
            
            return result_df
        except Exception as e:
            logger.error(f"填补缺失数据时出错: {str(e)}")
            return df
    
    def generate_analysis_report(self, df: pd.DataFrame, model: str = "deepseek-chat") -> str:
        """生成数据分析报告
        
        Args:
            df: 输入的DataFrame
            model: 模型名称，默认为"deepseek-chat"
            
        Returns:
            数据分析报告文本
        """
        try:
            if not self.api:
                return "无法生成分析报告：API客户端未初始化"
            
            # 简化一下数据以便于分析
            # 只保留关键列以减少token使用
            key_columns = []
            
            # 尝试找出关键列
            possible_key_columns = [
                '楼号', '房号', '姓名', '学工号', '入住天数', '合计天数', 
                '宿舍水电费', '个人水电费', '个人租金', '合计', '入住性质'
            ]
            
            for col in possible_key_columns:
                if col in df.columns:
                    key_columns.append(col)
            
            # 如果没有找到任何关键列，使用所有列
            if not key_columns:
                key_columns = df.columns.tolist()
            
            # 创建简化的DataFrame
            simplified_df = df[key_columns].copy()
            
            # 将DataFrame转换为字符串以供API使用
            df_str = simplified_df.head(20).to_string()
            
            # 构建提示
            prompt = f"""作为数据分析专家，请基于以下宿舍费用数据生成简短的分析报告（不超过600字）：

```
{df_str}
```

注意：上述数据只是前20行样本，总共有{len(df)}行数据。

请在报告中分析以下几点：
1. 数据的基本特征（如宿舍数量、人员数量等）
2. 费用分布情况（如平均水电费、租金分布等）
3. 是否存在异常数据
4. 对管理层有用的见解和建议

请使用markdown格式输出，保持简洁清晰。"""
            
            # 调用API获取分析报告
            report, _ = self.api.generate_response(prompt, model)
            
            return report
        except Exception as e:
            logger.error(f"生成分析报告时出错: {str(e)}")
            return f"无法生成分析报告: {str(e)}"
    
    def process_and_analyze(self, file_path: str, model: str = "deepseek-chat") -> Tuple[pd.DataFrame, str, str]:
        """处理Excel文件并生成分析报告
        
        Args:
            file_path: Excel文件路径
            model: 模型名称，默认为"deepseek-chat"
            
        Returns:
            处理后的DataFrame、处理消息和分析报告
        """
        try:
            # 处理Excel文件
            df = self.process_excel(file_path)
            
            # 计算费用
            df, message = self.calculate_fees_using_api(df)
            
            # 填补缺失数据
            df = self.fill_missing_data(df, model)
            
            # 生成分析报告
            analysis = self.generate_analysis_report(df, model)
            
            return df, message, analysis
        except Exception as e:
            logger.error(f"处理与分析文件时出错: {str(e)}")
            error_message = f"处理文件时出错: {str(e)}"
            return pd.DataFrame(), error_message, ""
    
    def _standardize_column_names(self, df: pd.DataFrame) -> None:
        """标准化列名
        
        Args:
            df: 需要标准化列名的DataFrame
        """
        # 创建列名映射字典
        column_mapping = {
            # 宿舍信息
            '宿舍': '楼号', '宿舍号': '楼号', '楼': '楼号', '楼宇': '楼号',
            '房间': '房号', '房间号': '房号', '宿舍房号': '房号',
            
            # 个人信息
            '学号': '学工号', '工号': '学工号', '教工号': '学工号',
            '姓名': '姓名', '人员': '姓名', '住宿人': '姓名',
            
            # 时间信息
            '月份': '月份',
            '入住时间': '入住天数', '住宿天数': '入住天数', '天数': '入住天数',
            '总天数': '合计天数', '月总天数': '合计天数',
            
            # 费用信息
            '水电费': '宿舍水电费', '宿舍水电': '宿舍水电费', '房间水电费': '宿舍水电费',
            '个人水电': '个人水电费',
            '租金': '租金', '房租': '租金', '宿舍租金': '租金',
            '个人租': '个人租金', '自费租金': '个人租金', '租金（个人）': '个人租金',
            '总费用': '合计', '总计': '合计', '费用合计': '合计'
        }
        
        # 应用映射
        for col in df.columns:
            for key, value in column_mapping.items():
                if key.lower() in col.lower():
                    # 如果找到匹配，重命名列
                    df.rename(columns={col: value}, inplace=True)
                    break

    def generate_table_from_chat(self, df: pd.DataFrame, user_request: str, chat_messages=None, all_dataframes=None) -> Tuple[Dict, str]:
        """根据用户聊天请求生成表格数据
        
        Args:
            df: 主数据框
            user_request: 用户请求内容
            chat_messages: 聊天历史记录
            all_dataframes: 所有上传的数据框列表，包括不匹配的文件
            
        Returns:
            表格JSON数据和消息
        """
        try:
            if df is None or df.empty:
                # 如果主数据框为空，但有其他数据框，尝试使用第一个非空数据框
                if all_dataframes and len(all_dataframes) > 0:
                    for df_info in all_dataframes:
                        if not df_info["dataframe"].empty:
                            df = df_info["dataframe"]
                            logger.info(f"主数据框为空，使用文件 '{df_info['filename']}' 作为数据源")
                            break
                
                # 如果还是没有可用数据，返回错误
                if df is None or df.empty:
                    return {"error": "没有可用的数据"}, "没有数据可以处理"
            
            # 限制样本数据量，避免提示词过长
            sample_rows = min(30, len(df))
            
            # 将DataFrame转换为字符串表示，更易于模型理解
            # df_string = df.head(sample_rows).to_string(index=False)
            
            # 收集数据样本和统计信息
            data_sample = []
            for _, row in df.head(sample_rows).iterrows():
                row_dict = {}
                for col, val in row.items():
                    # 确保所有值都可以JSON序列化
                    if pd.isna(val):
                        row_dict[col] = None
                    elif isinstance(val, (pd.Timestamp, np.datetime64)):
                        row_dict[col] = val.isoformat() if hasattr(val, 'isoformat') else str(val)
                    elif isinstance(val, (np.int64, np.float64)):
                        row_dict[col] = int(val) if isinstance(val, np.int64) else float(val)
                    else:
                        row_dict[col] = str(val)
                data_sample.append(row_dict)
            
            # 创建主数据统计信息
            data_stats = {
                "main_table": {
                    "columns": list(df.columns),
                    "row_count": len(df),
                    "numeric_columns": [str(col) for col in df.select_dtypes(include=np.number).columns],
                    "sample_data": data_sample
                }
            }
            
            # 如果有其他数据框，添加它们的信息
            if all_dataframes and len(all_dataframes) > 0:
                data_stats["all_tables"] = []
                
                for df_info in all_dataframes:
                    file_df = df_info["dataframe"]
                    filename = df_info["filename"]
                    
                    # 收集此数据框的样本
                    file_sample_rows = min(20, len(file_df))
                    file_data_sample = []
                    
                    for _, row in file_df.head(file_sample_rows).iterrows():
                        row_dict = {}
                        for col, val in row.items():
                            if pd.isna(val):
                                row_dict[col] = None
                            elif isinstance(val, (pd.Timestamp, np.datetime64)):
                                row_dict[col] = val.isoformat() if hasattr(val, 'isoformat') else str(val)
                            elif isinstance(val, (np.int64, np.float64)):
                                row_dict[col] = int(val) if isinstance(val, np.int64) else float(val)
                            else:
                                row_dict[col] = str(val)
                        file_data_sample.append(row_dict)
                    
                    # 添加此文件数据框的信息
                    table_info = {
                        "filename": filename,
                        "columns": list(file_df.columns),
                        "row_count": len(file_df),
                        "numeric_columns": [str(col) for col in file_df.select_dtypes(include=np.number).columns],
                        "sample_data": file_data_sample
                    }
                    
                    data_stats["all_tables"].append(table_info)
            
            # 添加提示以改善表格生成
            enhanced_prompt = f"{user_request}\n\n请基于提供的数据创建一个适当的表格，并包含以下信息：\n1. 表格名称\n2. 表格列头\n3. 表格数据行\n4. 表格摘要描述"
            
            # 如果有多个数据源，添加特殊提示
            if all_dataframes and len(all_dataframes) > 1:
                enhanced_prompt += "\n\n注意：用户可能想要基于特定文件或所有文件创建表格。请根据用户请求确定应该使用哪个文件的数据。"
            
            # 提供聊天历史上下文
            if chat_messages:
                chat_context = ""
                for msg in chat_messages[-5:]:  # 只使用最近5条消息
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        chat_context += f"{role}: {content}\n\n"
                
                if chat_context:
                    enhanced_prompt += f"\n\n聊天历史上下文:\n{chat_context}"
            
            # 调用API生成表格
            try:
                # 调用DeepSeek API生成表格
                if not self.api:
                    return {"error": "API客户端未初始化"}, "API客户端未初始化，无法生成表格"
                
                # 准备提示词
                system_prompt = """你是一个专业的宿舍数据分析助手，精通数据分析和表格生成。
                请根据用户的请求和提供的数据生成一个表格。
                
                你将获得一些数据样本，包括主数据表和可能的多个文件数据表。
                请分析用户的需求，确定应该基于哪个数据表生成结果。
                如果用户未明确指定，应该默认使用主数据表。
                
                请返回一个JSON对象，包含以下字段：
                {
                  "table_name": "表格名称",
                  "headers": ["列1", "列2", "列3"],
                  "data": [
                    ["行1列1值", "行1列2值", "行1列3值"],
                    ["行2列1值", "行2列2值", "行2列3值"]
                  ],
                  "summary": "这个表格的简要描述，说明表格包含的内容和目的"
                }
                
                仅返回JSON对象，不要添加其他文本。确保返回的是有效的JSON格式。
                """
                
                # 准备用户消息
                user_message = f"""请根据以下数据生成一个表格。

我需要的表格是：{enhanced_prompt}

数据统计信息：{json.dumps(data_stats, ensure_ascii=False, indent=2)}

请生成一个JSON对象，包含表格名称、表头、数据行和表格摘要。
请确保返回的是有效的JSON格式。
请分析我的请求，确定应该使用哪个数据源，并生成适当的表格。
"""
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
                
                # 调用API
                response, response_id = self.api.generate_response(user_message, model="deepseek-chat")
                
                # 尝试从回复中提取JSON
                try:
                    # 找到JSON对象的边界
                    json_start = response.find("{")
                    json_end = response.rfind("}")
                    
                    if json_start >= 0 and json_end > json_start:
                        json_text = response[json_start:json_end+1]
                        table_data = json.loads(json_text)
                        
                        # 验证必要的字段
                        required_fields = ["table_name", "headers", "data"]
                        if all(field in table_data for field in required_fields):
                            # 如果没有提供summary，添加一个默认的
                            if "summary" not in table_data:
                                table_data["summary"] = f"表格包含 {len(table_data['data'])} 行数据"
                                
                            return table_data, "成功使用API生成表格"
                        else:
                            missing = [f for f in required_fields if f not in table_data]
                            return {"error": f"API返回的JSON缺少必要字段: {missing}"}, "API返回的JSON缺少必要字段"
                    else:
                        # 没有找到有效的JSON，尝试备用方法
                        logger.error("API响应中没有找到有效的JSON")
                        return self.generate_table_fallback(df, user_request, all_dataframes)
                except json.JSONDecodeError as json_err:
                    logger.error(f"解析API返回的JSON时出错: {str(json_err)}")
                    return self.generate_table_fallback(df, user_request, all_dataframes)
                
            except Exception as api_error:
                logger.error(f"调用API生成表格时出错: {str(api_error)}")
                return self.generate_table_fallback(df, user_request, all_dataframes)
                
        except Exception as e:
            logger.error(f"生成表格时出错: {str(e)}")
            return {"error": str(e)}, f"生成表格时出错: {str(e)}"
            
    def generate_table_fallback(self, df: pd.DataFrame, user_request: str, all_dataframes=None) -> Tuple[Dict, str]:
        """当API生成表格失败时的后备方案
        
        Args:
            df: 主数据框
            user_request: 用户请求
            all_dataframes: 所有上传的数据框列表
            
        Returns:
            基本表格和消息
        """
        try:
            # 确定使用哪个数据框
            target_df = df
            target_file = "主数据表"
            
            # 如果主数据框为空或很小，尝试从其他数据框中选择
            if all_dataframes and (df is None or df.empty or len(df) < 5):
                # 找到最大的数据框
                max_rows = 0
                for df_info in all_dataframes:
                    curr_df = df_info["dataframe"]
                    if len(curr_df) > max_rows:
                        max_rows = len(curr_df)
                        target_df = curr_df
                        target_file = df_info["filename"]
            
            if target_df is None or target_df.empty:
                return {
                    "table_name": "空表格",
                    "headers": ["提示"],
                    "data": [["无可用数据"]],
                    "summary": "无法生成表格，因为没有可用数据"
                }, "无可用数据"
            
            # 准备一个简单的表格，使用前10行数据
            headers = list(target_df.columns)[:10]  # 最多取10列
            
            # 如果列数太多，添加一个说明
            headers_note = ""
            if len(target_df.columns) > 10:
                headers_note = f"(仅显示前10列，总共{len(target_df.columns)}列)"
            
            # 准备数据行
            data = []
            for _, row in target_df.head(20).iterrows():  # 最多取20行
                data_row = []
                for col in headers:
                    # 确保值是字符串
                    val = row.get(col, "")
                    if pd.isna(val):
                        data_row.append("")
                    else:
                        data_row.append(str(val))
                data.append(data_row)
            
            return {
                "table_name": f"数据表格 {target_file} {headers_note}",
                "headers": headers,
                "data": data,
                "summary": f"该表格展示了{target_file}中的数据样本，包含{len(data)}行。由于API生成失败，这是一个基本的表格展示。"
            }, "使用备用方案生成基本表格"
            
        except Exception as e:
            logger.error(f"生成备用表格时出错: {str(e)}")
            # 创建一个最小化的表格
            return {
                "table_name": "错误表格",
                "headers": ["错误类型", "错误详情"],
                "data": [["生成表格失败", str(e)]],
                "summary": "由于发生错误，无法生成正常表格"
            }, f"生成备用表格时出错: {str(e)}"
    
    def create_excel_from_json(self, table_data: Dict) -> Tuple[bytes, str]:
        """从JSON表格数据创建Excel文件
        
        Args:
            table_data: 表格JSON数据，包含headers和data字段
            
        Returns:
            Excel文件内容(bytes)和文件名
        """
        try:
            # 验证表格数据格式
            if "headers" not in table_data or "data" not in table_data:
                raise ValueError("无效的表格数据格式，缺少headers或data字段")
            
            # 创建DataFrame，确保字符串类型一致性
            df = pd.DataFrame(table_data["data"], columns=table_data["headers"])
            
            # 处理数据类型，避免混合类型错误
            for col in df.columns:
                # 检查列是否可以转换为数值
                try:
                    # 尝试转换为数值类型
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # 如果有NaN值，填充为0
                    if df[col].isna().any():
                        df[col] = df[col].fillna(0)
                except:
                    # 如果无法转换为数值，确保为字符串类型
                    df[col] = df[col].astype(str)
            
            # 生成文件名
            table_name = table_data.get("table_name", "表格数据")
            timestamp = int(time.time())
            filename = f"{table_name}_{timestamp}.xlsx"
            
            # 创建Excel文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                try:
                    # 首先尝试使用xlsxwriter引擎
                    logger.info("尝试使用xlsxwriter引擎创建Excel")
                    with pd.ExcelWriter(tmp.name, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name=table_name[:31])  # Excel工作表名称限制为31个字符
                except ImportError:
                    try:
                        # 如果没有xlsxwriter，尝试使用openpyxl
                        logger.info("xlsxwriter不可用，尝试使用openpyxl引擎")
                        with pd.ExcelWriter(tmp.name, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name=table_name[:31])
                    except ImportError:
                        # 如果两者都不可用，回退到最基本的引擎
                        logger.info("openpyxl也不可用，尝试使用基本引擎")
                        df.to_excel(tmp.name, index=False, sheet_name=table_name[:31])
                
                # 关闭临时文件
                tmp.close()
                
                try:
                    # 读取Excel文件内容
                    with open(tmp.name, 'rb') as f:
                        excel_data = f.read()
                    
                    # 删除临时文件
                    try:
                        os.unlink(tmp.name)
                    except:
                        logger.warning(f"无法删除临时文件: {tmp.name}")
                    
                    return excel_data, filename
                except Exception as read_error:
                    logger.error(f"读取Excel文件内容时出错: {str(read_error)}")
                    # 尝试使用CSV作为备选
                    csv_data = df.to_csv(index=False).encode('utf-8-sig')
                    csv_filename = f"{table_name}_{timestamp}.csv"
                    return csv_data, csv_filename
        
        except Exception as e:
            logger.error(f"创建Excel文件时出错: {str(e)}")
            
            # 创建一个备用的CSV文件
            try:
                logger.info("尝试创建CSV文件作为备用")
                # 如果表格数据可用，创建DataFrame
                if "headers" in table_data and "data" in table_data:
                    df = pd.DataFrame(table_data["data"], columns=table_data["headers"])
                    csv_data = df.to_csv(index=False).encode('utf-8-sig')
                    csv_filename = f"{table_data.get('table_name', '表格数据')}_{timestamp}.csv"
                    return csv_data, csv_filename
                else:
                    # 创建一个错误信息的CSV
                    df = pd.DataFrame([["创建Excel文件时出错", str(e)]], 
                                      columns=["错误类型", "错误详情"])
                    csv_data = df.to_csv(index=False).encode('utf-8-sig')
                    csv_filename = f"错误报告_{timestamp}.csv"
                    return csv_data, csv_filename
            except:
                # 如果所有尝试都失败，抛出原始错误
                raise ValueError(f"创建Excel文件时出错: {str(e)}")