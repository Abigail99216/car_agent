import time
import json
import pdfplumber
import torch
import transformers
import jieba
import sklearn
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


questions = json.load(open("questions.json"))

pdf = pdfplumber.open("初赛训练数据集.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(page_idx + 1),
        'content': pdf.pages[page_idx].extract_text()
    })

# 加载重排序模型
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
#rerank_model.cuda()
rerank_model.eval()

pdf_content_words = [jieba.lcut(x['content']) for x in pdf_content]
bm25 = BM25Okapi(pdf_content_words)

for query_idx in tqdm(range(len(questions))):
  doc_scores = bm25.get_scores(jieba.lcut(questions[query_idx]["question"]))
  #找到得分最高的4个页面索引
  max_score_page_idxs = doc_scores.argsort()[-4:]
  #将问题与这4个页面的内容组成对
  pairs = []
  for idx in max_score_page_idxs:
    pairs.append([questions[query_idx]["question"], pdf_content[idx]['content']])
    #编码，包括填充和截断，并转化为pytorch张量
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        inputs = {key: inputs[key] for key in inputs.keys()}
        #rerank_models重新排序，得到分数
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
        max_score_page_idx = max_score_page_idxs[scores.cpu().numpy().argmax()]
        questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx + 1)
        prompt = '''你是一个汽车专家，帮我结合给定的资料，回答一个问题。如果问题无法从资料中获得，请输出结合给定的资料，无法回答问题。
    资料：{0}

    问题：{1}
       '''.format(
        pdf_content[max_score_page_idx]['content'],
        questions[query_idx]["question"]
    )
    answer = ask_glm(prompt)['choices'][0]['message']['content']
    questions[query_idx]['answer'] = answer


st.set_page_config(page_title="汽车专家gpt")

with st.sidebar:
    st.title('汽车专家gpt')
    if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
        st.success('API Token已经配置', icon='✅')
        key = st.session_state['API_TOKEN']
    else:
        key = "c9bc35e8e7c1c076a8aaba862efb19af.DhiaibnU9Mys34de"

    key = st.text_input('输入Token:', type='password', value=key)

    st.session_state['API_TOKEN'] = key

    model = st.selectbox("选择模型", ["glm-3-turbo", "glm-4"])
    max_tokens = st.slider("max_tokens", 0, 2000, value=512)
    temperature = st.slider("temperature", 0.0, 2.0, value=0.8)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "你好我是ChatGLM，我可以为您提供汽车方面的信息"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if "dataframe" not in st.session_state.keys():
    uploaded_file = st.file_uploader("上传你需要分析的文件")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe.head(10))
        st.session_state["dataframe"] = dataframe
    else:
        dataframe = None
else:
    dataframe = st.session_state["dataframe"]

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "你好我是ChatGLM，有什么可以帮助你的？"}]

st.sidebar.button('清空聊天记录', on_click=clear_chat_history)
