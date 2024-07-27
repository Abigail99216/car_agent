__import__('pysqlite3') 
import time
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import sys
sys.path.append("../知识库") # 将父目录放入系统路径中
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import json
import pdfplumber
import torch
import transformers
import jieba
import sklearn
from pickle import NONE
import streamlit as st
import requests
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import jwt

#实际key,过期时间
def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


#api调用
def ask_glm(content):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
      'Content-Type': 'application/json',
      'Authorization': generate_token("c9bc35e8e7c1c076a8aaba862efb19af.DhiaibnU9Mys34de", 1000)
    }

    data = {
        "model": "glm-4",
        "messages": [{"role": "user", "content": content}]
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

def get_vectordb():
    #定义embeddings
    embedding = ZhipuAIEmbeddings()
    persist_directory = '../知识库/data_base/vector_db/chroma'
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    return vectordb

#不带历史记录
def get_chat_qa_chain(question:str):
    vectordb = get_vectordb()
    llm = ChatZhipuAI(model = "glm-4", temperature=0, zhipuai_api_key=c9bc35e8e7c1c076a8aaba862efb19af.DhiaibnU9Mys34de)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']


#带有历史记录
def get_qa_chain(question:str):
    vectordb = get_vectordb()
    llm = ChatZhipuAI(model = "glm-4", temperature=0, zhipuai_api_key=c9bc35e8e7c1c076a8aaba862efb19af.DhiaibnU9Mys34de)
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]

#streamlit
def main():
    st.title('汽车问答助手')
    if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
        st.success('API Token已经配置', icon='✅')
        key = st.session_state['API_TOKEN']
    else:
        key = "c9bc35e8e7c1c076a8aaba862efb19af.DhiaibnU9Mys34de"
        key = st.text_input('输入Token:', type='password', value=key)
        st.session_state['API_TOKEN'] = key
        
        # 添加一个选择按钮来选择不同的模型
    #selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])
    
    st.sidebar.button('清空聊天记录', on_click=clear_chat_history)

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "None":
            answer = ask_glm(prompt)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt)
        elif selected_method =="chat_qa_chain":
            answer = get_chat_qa_chain(prompt)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "你好我是ChatGLM，我可以为您提供汽车方面的信息"}]

if __name__ == "__main__":
    main()
