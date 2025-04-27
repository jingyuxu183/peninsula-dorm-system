import streamlit as st
import pandas as pd
import os
import tempfile
import time
from backend import DormDataProcessor
import numpy as np
from typing import List, Dict, Tuple
import logging
import sys
import subprocess
import webbrowser

# 尝试导入dotenv，如果不存在则提供提示但不终止程序
try:
    from dotenv import load_dotenv
    dotenv_available = True
except ImportError:
    dotenv_available = False

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从run.py合并的功能：检查环境变量
def check_env():
    """检查环境变量是否已配置"""
    if not dotenv_available:
        st.warning("⚠️ 缺少 python-dotenv 依赖：环境变量无法正确加载")
        st.info("您可以运行 `pip install python-dotenv` 安装此依赖")
        # 尝试直接从环境变量获取API密钥
        if not os.getenv("DEEPSEEK_API_KEY"):
            st.warning("⚠️ 未找到DEEPSEEK_API_KEY环境变量！请设置此环境变量。")
            st.info("您可以创建一个.env文件，并添加以下内容：\nDEEPSEEK_API_KEY=您的API密钥")
            return False
        return True
    
    # 如果dotenv可用，加载环境变量
    load_dotenv()
    if not os.getenv("DEEPSEEK_API_KEY"):
        st.warning("⚠️ 未找到DEEPSEEK_API_KEY环境变量！请在.env文件中设置。")
        st.info("您可以创建一个.env文件，并添加以下内容：\nDEEPSEEK_API_KEY=您的API密钥")
        return False
    return True

# 从run.py合并的功能：检查依赖
def check_dependencies():
    """检查所需依赖是否已安装（Streamlit版本）"""
    required = ["streamlit", "pandas", "openpyxl", "requests"]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    # 单独检查python-dotenv，因为前面已经处理过这个依赖
    if not dotenv_available:
        missing.append("python-dotenv")
    
    if missing:
        st.error(f"⚠️ 缺少以下依赖: {', '.join(missing)}")
        st.info(f"请安装所需依赖: pip install {' '.join(missing)}")
        return False
    return True

# 从run.py合并的功能：演示获取模型列表
def demo_list_models():
    """演示获取模型列表的代码（Streamlit版本）"""
    if not dotenv_available:
        st.error("⚠️ 缺少 python-dotenv 依赖，无法执行示例。")
        st.info("请运行 `pip install python-dotenv` 安装此依赖")
        return
        
    st.subheader("DeepSeek模型列表示例")
    
    st.code("""from openai import OpenAI

# 创建客户端
client = OpenAI(api_key="<您的API密钥>", base_url="https://api.deepseek.com")

# 获取模型列表
models = client.models.list()

# 打印模型信息
print("可用模型列表:")
for model in models.data:
    print(f"- {model.id}")""", language="python")
    
    if st.button("执行示例代码"):
        try:
            from openai import OpenAI
            
            # 获取API密钥
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                st.warning("⚠️ 未找到DEEPSEEK_API_KEY环境变量！")
                return
            
            # 创建客户端
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            
            # 获取模型列表
            try:
                models = client.models.list()
                
                # 打印模型信息
                st.success("获取模型列表成功!")
                st.write("可用模型列表:")
                for model in models.data:
                    st.write(f"- {model.id}")
            except Exception as e:
                st.error(f"⚠️ 获取模型列表失败: {str(e)}")
                st.info("请确保您的API密钥正确，并且网络连接正常。")
        except ImportError:
            st.error("⚠️ 未安装所需依赖，无法执行示例。")
            st.info("请确保已安装以下依赖: openai, python-dotenv")

# 设置页面配置
st.set_page_config(
    page_title="半岛智宿系统",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置自定义CSS样式
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .step-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #e0f7fa;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    h1, h2, h3 {
        color: #0e4a67;
    }
    .stButton>button {
        background-color: #0e4a67;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1976d2;
    }
    /* 自定义聊天容器 */
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-bottom: 10px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'processing_msg' not in st.session_state:
    st.session_state.processing_msg = ""
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = ""
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'processed_chat_messages' not in st.session_state:
    st.session_state.processed_chat_messages = []
if 'table_json' not in st.session_state:
    st.session_state.table_json = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "deepseek-chat"
if 'uploaded_files_info' not in st.session_state:
    st.session_state.uploaded_files_info = []  # 存储已上传文件的信息
if 'all_dataframes' not in st.session_state:
    st.session_state.all_dataframes = []  # 存储所有上传的数据框，包括列不匹配的

# 初始化宿舍数据处理器
processor = DormDataProcessor()

# 应用标题
st.title("半岛智宿系统")
st.markdown("---")

# 处理重复列函数
def clean_dataframe(df):
    if df is None:
        return None
    
    # 查找形如"X.1"的重复列名
    duplicate_cols = []
    for col in df.columns:
        if '.' in col and col.split('.')[-1].isdigit():
            base_col = col.split('.')[0]
            if base_col in df.columns:
                duplicate_cols.append(col)
    
    # 删除重复列
    if duplicate_cols:
        df = df.drop(columns=duplicate_cols)
    
    return df

# 格式化DataFrame为清晰的表格形式
def format_dataframe(df, max_rows=10):
    """将DataFrame格式化为清晰易读的表格形式"""
    if len(df) == 0:
        return "空数据表"
    
    # 限制行数
    display_df = df.head(max_rows)
    
    # 获取列名
    headers = list(display_df.columns)
    
    # 格式化表头
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["-" * len(col) for col in headers]) + " |"
    
    # 格式化数据行
    data_rows = []
    for _, row in display_df.iterrows():
        formatted_row = "| " + " | ".join([str(row[col]) for col in headers]) + " |"
        data_rows.append(formatted_row)
    
    # 组合成表格
    table = [header_row, separator] + data_rows
    
    # 添加行数信息
    rows_info = f"\n\n总行数: {len(df)} 行，显示前 {min(max_rows, len(df))} 行"
    
    return "\n".join(table) + rows_info

# 第一步：上传Excel文件
with st.container():
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.subheader("第一步：上传Excel文件")
    
    # 修改为支持多文件上传，最多5个文件
    uploaded_files = st.file_uploader("选择宿舍数据Excel文件（可多选，最多5个）", type=["xlsx", "xls"], accept_multiple_files=True)
    
    # 限制文件数量
    if len(uploaded_files) > 5:
        st.warning("⚠️ 最多只能上传5个文件，将只处理前5个文件")
        uploaded_files = uploaded_files[:5]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_files and st.button("读取文件内容"):
            with st.spinner("正在读取文件..."):
                # 合并后的数据框
                combined_df = None
                
                # 清除之前的上传文件信息和数据框
                st.session_state.uploaded_files_info = []
                st.session_state.all_dataframes = []
                
                # 处理每个上传的文件
                for uploaded_file in uploaded_files:
                    # 创建临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_path = tmp.name
                    
                    try:
                        # 读取原始文件，但不计算费用
                        file_df = processor.process_excel(temp_path, calculate=False)
                        
                        # 清理重复列
                        file_df = clean_dataframe(file_df)
                        
                        # 记录文件信息
                        file_info = {
                            "filename": uploaded_file.name,
                            "rows": len(file_df),
                            "columns": list(file_df.columns)
                        }
                        st.session_state.uploaded_files_info.append(file_info)
                        
                        # 保存每个文件的数据框 - 新增的功能
                        st.session_state.all_dataframes.append({
                            "filename": uploaded_file.name,
                            "dataframe": file_df
                        })
                        
                        # 将数据合并到主数据框
                        if combined_df is None:
                            combined_df = file_df
                        else:
                            # 尝试合并数据框，处理可能的列不匹配问题
                            try:
                                # 检查列名是否一致
                                if set(combined_df.columns) == set(file_df.columns):
                                    # 列名完全一致，直接合并
                                    combined_df = pd.concat([combined_df, file_df], ignore_index=True)
                                else:
                                    # 列名不完全一致，只合并共同的列
                                    common_cols = list(set(combined_df.columns) & set(file_df.columns))
                                    if common_cols:
                                        # 至少有一些共同列，合并这些列
                                        combined_df = pd.concat(
                                            [combined_df[common_cols], file_df[common_cols]], 
                                            ignore_index=True
                                        )
                                        st.info(f"ℹ️ 文件 '{uploaded_file.name}' 的列与其他文件不完全匹配，只合并了共同的列，但文件已完整读取")
                                    else:
                                        # 没有共同列，但仍保存文件数据
                                        st.info(f"ℹ️ 文件 '{uploaded_file.name}' 的列与其他文件完全不匹配，不会合并到主数据表，但文件已完整读取")
                            except Exception as merge_err:
                                st.warning(f"合并文件 '{uploaded_file.name}' 时出错: {str(merge_err)}，但文件已完整读取")
                        
                    except Exception as e:
                        st.error(f"读取文件 '{uploaded_file.name}' 时出错: {str(e)}")
                    
                    # 删除临时文件
                    os.unlink(temp_path)
                
                if combined_df is not None:
                    # 更新会话状态
                    st.session_state.original_df = combined_df
                    st.session_state.processed_df = None  # 清除之前的处理结果
                    st.session_state.processing_msg = ""
                    st.session_state.analysis_result = ""
                    
                    # 清空之前的聊天记录
                    st.session_state.chat_messages = []
                    st.session_state.processed_chat_messages = []
                    
                    # 构建文件信息摘要
                    files_summary = ""
                    for i, file_info in enumerate(st.session_state.uploaded_files_info):
                        files_summary += f"📄 {i+1}. {file_info['filename']}: {file_info['rows']}行数据，{len(file_info['columns'])}列\n"
                    
                    # 构建初始系统消息
                    initial_message = f"""您好！我已成功读取您上传的{len(st.session_state.uploaded_files_info)}个文件。

已处理的文件：
{files_summary}

主数据表包含{len(combined_df)}条记录。
您可以向我询问关于任何已上传文件的问题。"""

                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": initial_message
                    })
                    
                    # 提供所有数据框的上下文信息
                    all_dfs_context = ""
                    for i, df_info in enumerate(st.session_state.all_dataframes):
                        filename = df_info["filename"]
                        df = df_info["dataframe"]
                        # 最多15行样本
                        sample_rows = min(10, len(df))
                        df_sample = format_dataframe(df, sample_rows)
                        all_dfs_context += f"\n文件 {i+1}: {filename}\n{df_sample}\n\n"
                    
                    # 提供主数据框的详细统计信息
                    try:
                        df_info = pd.DataFrame({
                            '统计量': combined_df.describe().index,
                            **{col: combined_df[col].describe() for col in combined_df.select_dtypes(include=np.number).columns[:5]}  # 限制为前5个数值列
                        })
                        df_info_formatted = format_dataframe(df_info)

                        # 获取实际数据样本，不仅仅是统计摘要
                        sample_rows = min(15, len(combined_df))  # 最多15行样本
                        df_sample_formatted = format_dataframe(combined_df, sample_rows)

                        # 获取列名详细列表
                        column_info = []
                        for col in combined_df.columns:
                            dtype = str(combined_df[col].dtype)
                            unique_values = combined_df[col].nunique()
                            sample_val = str(combined_df[col].iloc[0])
                            if len(sample_val) > 30:  # 如果样本值太长，截断它
                                sample_val = sample_val[:30] + "..."
                            column_info.append(f"- {col}: {dtype} (样本值: {sample_val}, 唯一值数量: {unique_values})")

                        columns_description = "\n".join(column_info)

                        data_context = f"""
                        合并后的主数据表:
                        {df_info_formatted}
                        
                        主数据表样本（前{sample_rows}行）:
                        {df_sample_formatted}
                        
                        主数据表列信息:
                        {columns_description}
                        
                        当前主数据表有{len(combined_df)}条记录，来自{len(st.session_state.uploaded_files_info)}个文件。
                        
                        -------- 所有上传文件的数据样本 --------
                        {all_dfs_context}
                        """
                    except Exception as e:
                        # 如果主数据框统计失败，只提供所有文件的样本
                        data_context = f"""
                        所有上传文件的数据样本:
                        {all_dfs_context}
                        
                        注意: 生成主数据表统计信息时出错: {str(e)}
                        """
                    
                    # 将数据背景作为系统消息添加供后续API调用使用
                    st.session_state.data_context = data_context
                    
                    st.success(f"成功读取{len(st.session_state.uploaded_files_info)}个文件，合并后主数据表包含{len(combined_df)}条记录")
                elif len(st.session_state.all_dataframes) > 0:
                    # 即使主数据框为空，但仍有读取到文件，也可以继续
                    st.session_state.original_df = pd.DataFrame()  # 创建空的主数据框
                    st.session_state.processed_df = None
                    st.session_state.processing_msg = ""
                    st.session_state.analysis_result = ""
                    
                    # 清空之前的聊天记录
                    st.session_state.chat_messages = []
                    st.session_state.processed_chat_messages = []
                    
                    # 构建文件信息摘要
                    files_summary = ""
                    for i, file_info in enumerate(st.session_state.uploaded_files_info):
                        files_summary += f"📄 {i+1}. {file_info['filename']}: {file_info['rows']}行数据，{len(file_info['columns'])}列\n"
                    
                    # 构建初始系统消息
                    initial_message = f"""您好！我已成功读取您上传的{len(st.session_state.uploaded_files_info)}个文件。

已处理的文件：
{files_summary}

由于文件结构差异较大，未创建合并数据表，但您可以询问关于任何已上传文件的问题。"""

                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": initial_message
                    })
                    
                    # 提供所有数据框的上下文信息
                    all_dfs_context = ""
                    for i, df_info in enumerate(st.session_state.all_dataframes):
                        filename = df_info["filename"]
                        df = df_info["dataframe"]
                        # 最多10行样本
                        sample_rows = min(10, len(df))
                        df_sample = format_dataframe(df, sample_rows)
                        all_dfs_context += f"\n文件 {i+1}: {filename}\n{df_sample}\n\n"
                    
                    data_context = f"""
                    所有上传文件的数据样本:
                    {all_dfs_context}
                    """
                    
                    # 将数据背景作为系统消息添加供后续API调用使用
                    st.session_state.data_context = data_context
                    
                    st.success(f"成功读取{len(st.session_state.uploaded_files_info)}个文件，但由于结构差异较大，未创建合并数据表")
                else:
                    st.error("没有成功读取任何有效数据，请检查上传的文件")
    
    # 显示已上传文件列表
    if st.session_state.uploaded_files_info:
        with st.expander("查看已处理文件详情"):
            for i, file_info in enumerate(st.session_state.uploaded_files_info):
                st.write(f"📄 {i+1}. **{file_info['filename']}**")
                st.write(f"   - 行数: {file_info['rows']}")
                st.write(f"   - 列数: {len(file_info['columns'])}")
                st.write(f"   - 列名: {', '.join(file_info['columns'][:5])}{'...' if len(file_info['columns']) > 5 else ''}")
                
    st.markdown('</div>', unsafe_allow_html=True)

# 第二步：显示原始数据和聊天界面
if st.session_state.original_df is not None:
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.subheader("第二步：查看原始数据")
        
        # 显示原始数据，仅显示前20行
        st.dataframe(st.session_state.original_df.head(20), use_container_width=True)
        
        # 创建聊天机器人界面
        st.markdown("### 数据助手")
        st.markdown("您可以向数据助手询问关于原始数据的问题，助手会记住上下文并支持多轮对话。您还可以要求生成表格并下载。")
        st.markdown("_注意：对话开始时助手只会简单显示已读取文件的信息，而不展示数据样本，以保持界面简洁。_")
        
        # 使用Streamlit原生聊天组件显示历史消息
        for i, message in enumerate(st.session_state.chat_messages):
            with st.chat_message(message["role"]):
                # 显示消息内容
                st.markdown(message["content"])
        
        # 使用Streamlit原生聊天输入组件
        if user_input := st.chat_input("输入您的问题或请求生成表格", key="chat_input_field"):
            # 添加用户消息到聊天历史
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_input
            })
            
            # 显示用户消息
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # 检查是否是表格生成请求
            is_table_request = False
            table_keywords = ["表格", "生成表", "创建表", "制作表", "汇总表", "导出表", "统计表", "分析表", 
                             "表单", "电子表格", "excel", "转为表格", "做个表", "做一个表", "数据表", "列表"]

            # 检查用户输入是否包含表格相关关键词
            for keyword in table_keywords:
                if keyword in user_input.lower():
                    is_table_request = True
                    break

            # 更智能的分析：检查是否是表格生成意图
            if not is_table_request:
                # 检查是否包含汇总、导出、整理等与表格相关的动作词
                action_keywords = ["汇总", "导出", "整理", "归纳", "分组", "分类", "统计", "计算总和", 
                                  "计算平均", "排序", "筛选", "结构化", "可视化"]
                data_keywords = ["数据", "信息", "记录", "费用", "金额", "人员", "宿舍", "楼号", "房间"]
                
                action_match = any(keyword in user_input for keyword in action_keywords)
                data_match = any(keyword in user_input for keyword in data_keywords)
                
                # 如果同时匹配动作词和数据词，很可能是表格请求
                if action_match and data_match:
                    is_table_request = True
            
            # 构建消息历史用于API请求
            messages = [
                {"role": "system", "content": "你是一个专业的数据分析助手，熟悉宿舍费用数据。请用简体中文回答用户问题。"}
            ]
            
            # 添加历史消息，但确保总数不超过10条以控制上下文长度
            chat_history = []
            for msg in st.session_state.chat_messages[-10:]:
                history_msg = {"role": msg["role"], "content": msg["content"]}
                chat_history.append(history_msg)
            
            # 将历史消息添加到API请求
            messages.extend(chat_history)
            
            # 添加数据背景信息到系统消息
            if hasattr(st.session_state, 'data_context') and st.session_state.data_context:
                messages[0]["content"] += f"\n\n{st.session_state.data_context}"
            else:
                # 如果没有存储的数据上下文，则生成一个简化版本
                # 提供数据背景信息
                df_info = pd.DataFrame({
                    '统计量': st.session_state.original_df.describe().index,
                    **{col: st.session_state.original_df[col].describe() for col in st.session_state.original_df.select_dtypes(include=np.number).columns[:5]}  # 限制为前5个数值列
                })
                df_info_formatted = format_dataframe(df_info)

                # 获取实际数据样本，不仅仅是统计摘要
                sample_rows = min(15, len(st.session_state.original_df))  # 最多15行样本
                df_sample_formatted = format_dataframe(st.session_state.original_df, sample_rows)

                # 获取列名详细列表
                column_info = []
                for col in st.session_state.original_df.columns:
                    dtype = str(st.session_state.original_df[col].dtype)
                    unique_values = st.session_state.original_df[col].nunique()
                    sample_val = str(st.session_state.original_df[col].iloc[0])
                    if len(sample_val) > 30:  # 如果样本值太长，截断它
                        sample_val = sample_val[:30] + "..."
                    column_info.append(f"- {col}: {dtype} (样本值: {sample_val}, 唯一值数量: {unique_values})")

                columns_description = "\n".join(column_info)

                data_context = f"""
                数据摘要:
                {df_info_formatted}
                
                实际数据样本（前{sample_rows}行）:
                {df_sample_formatted}
                
                列信息详细:
                {columns_description}
                
                当前有{len(st.session_state.original_df)}条数据记录。
                """
                
                # 将数据背景作为系统消息添加到最前面
                messages[0]["content"] += f"\n\n{data_context}"
            
            # 显示助手正在输入的状态
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("思考中...")
                
                try:
                    # 获取用户选择的模型
                    selected_model = st.session_state.selected_model
                    
                    # 如果是表格生成请求，使用表格生成功能
                    if is_table_request:
                        message_placeholder.markdown("正在生成表格数据...")
                        
                        # 传递所有文件数据框 - 新增内容
                        # 生成表格JSON数据，传递聊天历史记录和所有读取的文件数据
                        table_json, msg = processor.generate_table_from_chat(
                            st.session_state.original_df, 
                            user_input,
                            chat_messages=st.session_state.chat_messages,
                            all_dataframes=st.session_state.all_dataframes  # 传递所有数据框信息
                        )
                        
                        if "error" in table_json:
                            content = f"抱歉，生成表格时出错: {table_json['error']}"
                        else:
                            # 生成表格预览和下载选项
                            preview_text = f"""### {table_json.get('table_name', '数据表格')}

**表格描述**：{table_json.get('summary', '表格数据')}

**表格预览**："""

                            # 显示表格头部预览
                            headers = table_json.get("headers", [])
                            data_rows = table_json.get("data", [])

                            # 创建表格预览
                            if headers and data_rows:
                                # 创建markdown表格
                                table_md = "| " + " | ".join(headers) + " |\n"
                                table_md += "| " + " | ".join(["---" for _ in headers]) + " |\n"
                                
                                # 显示所有数据行
                                for i in range(len(data_rows)):
                                    # 确保所有单元格内容为字符串并限制长度
                                    row_data = []
                                    for cell in data_rows[i]:
                                        cell_str = str(cell) if cell is not None else ""
                                        # 如果单元格内容太长，截断它
                                        if len(cell_str) > 20:
                                            cell_str = cell_str[:17] + "..."
                                        row_data.append(cell_str)
                                    
                                    table_md += "| " + " | ".join(row_data) + " |\n"
                                
                                preview_text += f"\n{table_md}\n\n"
                            else:
                                if headers:
                                    # 如果有表头但没有数据行，显示空表格
                                    table_md = "| " + " | ".join(headers) + " |\n"
                                    table_md += "| " + " | ".join(["---" for _ in headers]) + " |\n"
                                    table_md += "| " + " | ".join(["" for _ in headers]) + " |\n"
                                    preview_text += f"\n{table_md}\n\n*表格暂无数据*\n\n"
                                else:
                                    preview_text += "\n\n*表格数据为空或格式不正确*\n\n"

                            # 添加表格统计信息
                            preview_text += f"""**表格统计信息**：
                            - 总行数：{len(data_rows)}
                            - 总列数：{len(headers)}
                            - 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}

                            您可以在下方下载完整表格。"""

                            # 设置会话状态保存表格数据供下载
                            st.session_state.table_json = table_json

                            content = preview_text
                        
                        # 显示最终回答
                        message_placeholder.markdown(content)
                        
                        # 添加助手回复到聊天历史
                        st.session_state.chat_messages.append({
                            "role": "assistant", 
                            "content": content
                        })
                    else:
                        # 普通对话请求，处理流式和非流式对话
                        try:
                            # 添加所有文件的上下文信息到系统消息 - 修改内容
                            if hasattr(st.session_state, 'data_context') and st.session_state.data_context:
                                # 确认messages第一条是系统消息
                                if messages[0]["role"] == "system":
                                    # 添加数据上下文
                                    messages[0]["content"] += f"\n\n{st.session_state.data_context}"
                                    
                                    # 添加关于查询多个文件的额外提示
                                    if len(st.session_state.all_dataframes) > 1:
                                        messages[0]["content"] += "\n\n请特别注意：用户可能在询问任何一个已上传的文件，请根据问题内容确定用户关注的是哪个文件的数据，或者综合分析所有文件的数据。"
                                
                            # 使用deepseek-chat模型
                            model = "deepseek-chat"
                            
                            content, _ = processor.api.stream_chat_completion(messages, model)
                            
                            # 显示最终回答
                            message_placeholder.markdown(content)
                            
                            # 添加助手回复到聊天历史
                            st.session_state.chat_messages.append({
                                "role": "assistant", 
                                "content": content
                            })
                        except Exception as e:
                            # 所有尝试都失败
                            content = f"抱歉，处理您的问题时出错: {str(e)}"
                            message_placeholder.markdown(content)
                            st.session_state.chat_messages.append({
                                "role": "assistant", 
                                "content": content
                            })
                
                except Exception as e:
                    # 所有尝试都失败，显示错误消息
                    error_msg = f"抱歉，处理您的问题时出错: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
        
        # 提供表格下载功能
        if 'table_json' in st.session_state and st.session_state.table_json:
            st.markdown("### 下载生成的表格")
            
            try:
                # 从JSON创建Excel文件
                try:
                    excel_data, filename = processor.create_excel_from_json(st.session_state.table_json)
                    
                    # 检查文件扩展名确定文件类型
                    is_excel = filename.lower().endswith('.xlsx')
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if is_excel else "text/csv"
                    
                    # 显示下载按钮
                    st.download_button(
                        label=f"下载{st.session_state.table_json.get('table_name', '数据表格')}",
                        data=excel_data,
                        file_name=filename,
                        mime=mime_type
                    )
                    
                    if not is_excel:
                        st.info("注意：由于系统限制，表格以CSV格式下载。请安装xlsxwriter库以启用Excel下载。")
                except Exception as excel_error:
                    st.error(f"Excel生成失败: {str(excel_error)}，将提供CSV下载选项")
                    
                # 创建一个基本的CSV备选方案
                if "headers" in st.session_state.table_json and "data" in st.session_state.table_json:
                    # 确保data是有效的二维数组
                    data = st.session_state.table_json["data"]
                    headers = st.session_state.table_json["headers"]
                    
                    # 处理data为空的情况
                    if not data or len(data) == 0:
                        # 创建一个只有表头的空DataFrame
                        df = pd.DataFrame(columns=headers)
                    else:
                        # 检查每行是否有足够的列
                        clean_data = []
                        for row in data:
                            # 如果行数据是None，创建空行
                            if row is None:
                                clean_data.append([None] * len(headers))
                            # 处理空列表的情况
                            elif len(row) == 0:
                                clean_data.append([None] * len(headers))
                            # 确保行长度与表头匹配
                            elif len(row) < len(headers):
                                # 填充缺少的单元格为None
                                padded_row = row + [None] * (len(headers) - len(row))
                                clean_data.append(padded_row)
                            else:
                                clean_data.append(row)
                                
                        df = pd.DataFrame(clean_data, columns=headers)
                        
                    # 处理数据中的None值
                    df = df.fillna("")
                    
                    csv_data = df.to_csv(index=False).encode('utf-8-sig')
                    table_name = st.session_state.table_json.get('table_name', '数据表格')
                    timestamp = int(time.time())
                    
                    st.download_button(
                        label=f"下载{table_name} (CSV格式)",
                        data=csv_data,
                        file_name=f"{table_name}_{timestamp}.csv",
                        mime="text/csv"
                    )
                
                # 添加查看详情的展开部分
                with st.expander("查看表格详细数据"):
                    st.json(st.session_state.table_json)
                    
                    # 如果有相当数量的数据，还可以显示为DataFrame
                    if "headers" in st.session_state.table_json and "data" in st.session_state.table_json:
                        # 确保data是有效的二维数组
                        data = st.session_state.table_json["data"]
                        headers = st.session_state.table_json["headers"]
                        
                        try:
                            # 处理data为空的情况
                            if not data or len(data) == 0:
                                # 创建一个只有表头的空DataFrame
                                df = pd.DataFrame(columns=headers)
                                st.info("表格数据为空")
                            else:
                                # 检查每行是否有足够的列
                                clean_data = []
                                for row in data:
                                    # 如果行数据是None，创建空行
                                    if row is None:
                                        clean_data.append([None] * len(headers))
                                    # 处理空列表的情况
                                    elif len(row) == 0:
                                        clean_data.append([None] * len(headers))
                                    # 确保行长度与表头匹配
                                    elif len(row) < len(headers):
                                        # 填充缺少的单元格为None
                                        padded_row = row + [None] * (len(headers) - len(row))
                                        clean_data.append(padded_row)
                                    else:
                                        clean_data.append(row)
                                        
                                df = pd.DataFrame(clean_data, columns=headers)
                            
                            # 处理数据中的None值
                            df = df.fillna("")
                            
                            st.dataframe(df, use_container_width=True)
                        except Exception as df_error:
                            st.error(f"显示表格数据时出错: {str(df_error)}")
                            st.write("原始数据:")
                            st.write(st.session_state.table_json)
            except Exception as e:
                st.error(f"准备表格下载时出错: {str(e)}")
                # 最后的备选方案 - 直接显示数据
                st.write("表格数据：")
                st.write(st.session_state.table_json)
        
        st.markdown('</div>', unsafe_allow_html=True)

# 第三步：显示处理结果
if st.session_state.processed_df is not None:
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.subheader("第三步：查看处理结果")
        
        # 显示处理消息
        if st.session_state.processing_msg:
            st.markdown(f'<div class="info-box">{st.session_state.processing_msg}</div>', unsafe_allow_html=True)
        
        # 显示处理后的数据
        st.dataframe(st.session_state.processed_df, use_container_width=True)
        
        # 创建处理结果聊天机器人界面
        st.markdown("### 结果分析助手")
        st.markdown("您可以向结果分析助手询问关于处理结果的问题，助手会记住上下文并支持多轮对话。")
        
        # 使用Streamlit原生聊天组件显示历史消息
        for i, message in enumerate(st.session_state.processed_chat_messages):
            with st.chat_message(message["role"]):
                # 显示消息内容
                st.markdown(message["content"])
        
        # 使用Streamlit原生聊天输入组件
        if processed_input := st.chat_input("输入您的问题", key="processed_chat_input_field"):
            # 添加用户消息到聊天历史
            st.session_state.processed_chat_messages.append({
                "role": "user", 
                "content": processed_input
            })
            
            # 显示用户消息
            with st.chat_message("user"):
                st.markdown(processed_input)
            
            # 构建消息历史用于API请求
            messages = [
                {"role": "system", "content": "你是一个专业的数据分析助手，熟悉宿舍费用计算结果数据。请用简体中文回答用户问题。"}
            ]
            
            # 添加历史消息，但确保总数不超过10条以控制上下文长度
            chat_history = []
            for msg in st.session_state.processed_chat_messages[-10:]:
                history_msg = {"role": msg["role"], "content": msg["content"]}
                chat_history.append(history_msg)
            
            # 将历史消息添加到API请求
            messages.extend(chat_history)
            
            # 提供数据背景信息
            df_info = st.session_state.processed_df.describe().to_string()
            # 获取实际数据样本，不仅仅是统计摘要
            sample_rows = min(15, len(st.session_state.processed_df))  # 最多15行样本
            df_sample = st.session_state.processed_df.head(sample_rows).to_string()
            
            # 获取列名详细列表
            column_info = []
            for col in st.session_state.processed_df.columns:
                unique_values = st.session_state.processed_df[col].nunique()
                sample_val = str(st.session_state.processed_df[col].iloc[0])
                if len(sample_val) > 30:  # 如果样本值太长，截断它
                    sample_val = sample_val[:30] + "..."
                column_info.append(f"- {col}: {st.session_state.processed_df[col].dtype} (样本值: {sample_val}, 唯一值数量: {unique_values})")
            
            columns_description = "\n".join(column_info)
            
            data_context = f"""
            数据摘要:
            {df_info}
            
            实际数据样本（前{sample_rows}行）:
            {df_sample}
            
            列信息详细:
            {columns_description}
            
            当前有{len(st.session_state.processed_df)}条数据记录。
            计算规则说明:
            - 个人水电费 = 入住天数 ÷ 合计天数 × 宿舍水电费
            - 个人租金（自费人员）= 租金 ÷ 床位数 × 入住天数 ÷ 合计天数
            - 合计 = 个人水电费 + 个人租金
            """
            
            # 将数据背景作为系统消息添加到最前面
            messages[0]["content"] += f"\n\n{data_context}"
            
            # 显示助手正在输入的状态
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("思考中...")
                
                try:
                    # 使用deepseek-chat模型
                    content, _ = processor.stream_chat_completion(messages, "deepseek-chat")
                    
                    # 显示最终回答
                    message_placeholder.markdown(content)
                    
                    # 添加助手回复到聊天历史
                    st.session_state.processed_chat_messages.append({
                        "role": "assistant", 
                        "content": content
                    })
                    
                except Exception as e:
                    # 所有尝试都失败，显示错误消息
                    error_msg = f"抱歉，处理您的问题时出错: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.processed_chat_messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
        
        # 提供下载功能
        st.markdown("### 下载生成的表格")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            csv = st.session_state.processed_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="下载CSV结果",
                data=csv,
                file_name="宿舍费用计算结果.csv",
                mime="text/csv"
            )
        
        with col2:
            try:
                # 创建输出缓冲区
                buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                buffer.close()
                
                # 写入Excel文件，尝试多种引擎
                try:
                    # 首先尝试xlsxwriter引擎
                    with pd.ExcelWriter(buffer.name, engine='xlsxwriter') as writer:
                        st.session_state.processed_df.to_excel(writer, index=False, sheet_name='计算结果')
                except ImportError:
                    try:
                        # 然后尝试openpyxl引擎
                        with pd.ExcelWriter(buffer.name, engine='openpyxl') as writer:
                            st.session_state.processed_df.to_excel(writer, index=False, sheet_name='计算结果')
                    except ImportError:
                        # 最后使用默认引擎
                        st.session_state.processed_df.to_excel(buffer.name, index=False, sheet_name='计算结果')
                
                # 读取Excel文件内容
                with open(buffer.name, 'rb') as f:
                    excel_data = f.read()
                
                # 删除临时文件
                try:
                    os.unlink(buffer.name)
                except:
                    pass  # 忽略删除错误
                
                # 显示下载按钮
                st.download_button(
                    label="下载Excel结果",
                    data=excel_data,
                    file_name="宿舍费用计算结果.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                # 如果Excel生成失败，提供CSV下载选项
                st.error(f"Excel生成失败: {str(e)}，将提供CSV下载选项")
                csv = st.session_state.processed_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="下载CSV结果 (Excel生成失败)",
                    data=csv,
                    file_name="宿舍费用计算结果.csv",
                    mime="text/csv"
                )
        
        # 显示自费人员费用汇总
        if '入住性质' in st.session_state.processed_df.columns:
            try:
                # 查找自费人员
                self_paying_rows = st.session_state.processed_df['入住性质'].astype(str).str.contains('自费')
                
                if self_paying_rows.any():
                    st.markdown("### 自费人员费用汇总")
                    
                    # 筛选自费人员数据
                    self_paying_df = st.session_state.processed_df[self_paying_rows].copy()
                    
                    # 计算汇总信息
                    total_people = len(self_paying_df)
                    total_utility = self_paying_df['个人水电费'].sum()
                    total_rent = self_paying_df['个人租金'].sum()
                    total_amount = self_paying_df['合计'].sum()
                    
                    # 显示汇总信息
                    st.markdown(f'<div class="success-box">'
                                f'<strong>自费人员总数:</strong> {total_people} 人<br>'
                                f'<strong>水电费总计:</strong> {total_utility:.2f} 元<br>'
                                f'<strong>租金总计:</strong> {total_rent:.2f} 元<br>'
                                f'<strong>费用总计:</strong> {total_amount:.2f} 元'
                                f'</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"计算自费人员汇总时出错: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# 第四步：数据分析
if st.session_state.analysis_result:
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.subheader("第四步：数据分析")
        
        # 显示分析结果
        st.markdown(st.session_state.analysis_result)
        st.markdown('</div>', unsafe_allow_html=True)

# 页脚版权信息
st.markdown("---")
st.markdown("© 2025 半岛智宿系统 v1.0", help="宿舍费用计算和分析工具")

# 用户指南
with st.sidebar:
    st.header("使用指南")
    
    # 执行环境和依赖检查
    env_ok = check_env()
    deps_ok = check_dependencies()
    
    if not env_ok or not deps_ok:
        st.error("⚠️ 环境检查发现问题，请查看上方提示。")
    
    # 添加模型列表示例选项
    if st.checkbox("查看DeepSeek模型列表示例"):
        demo_list_models()
    
    st.markdown("---")
    
    st.markdown("""
    ### 工具说明
    半岛智宿系统通过智能对话方式处理宿舍数据，主要功能包括:
    - 支持多文件上传（最多5个Excel文件）
    - 智能处理不同结构的数据文件
    - 通过对话分析任意已上传的文件
    - 生成自定义表格并导出
    - 支持跨文件数据分析和汇总
    
    ### 使用步骤
    1. 上传一个或多个宿舍数据Excel文件
    2. 查看原始数据和文件详情
    3. 与数据助手对话询问任意文件的信息
    4. 请求生成表格，支持所有已上传文件的数据
    5. 下载生成的Excel表格
    
    ### 多文件处理特性
    - 自动识别并合并相同结构的文件
    - 支持完全不同结构的多个文件并行分析
    - 可以针对特定文件或所有文件提问
    - 跨文件数据比较和整合
    
    ### 表格生成提示示例
    - "请生成一个宿舍水电费汇总表"
    - "分析并对比所有文件中的费用数据"
    - "根据第一个文件创建用户名单，根据第二个文件添加费用信息"
    - "统计不同宿舍楼的人员和费用分布情况"
    """)
    
    st.markdown("© 2025 半岛智宿系统 v1.0") 