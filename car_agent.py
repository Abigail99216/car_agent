import json
import pdfplumber
import torch
import transformers
import jieba
import sklearn
import streamlit as st


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

with st.sidebar:
    st.title('金融专家gpt')
    st.write('支持的大模型包括ChatGLM3和4')
    # 初始化的对话
    if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "你好我是ChatGLM，有什么可以帮助你的？"}]
    
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "你好我是ChatGLM，有什么可以帮助你的？"}]
    
st.sidebar.button('清空聊天记录', on_click=clear_chat_history)

# Streamlit 应用程序界面
def main():
    st.title('金融专家助手')
    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    messages = st.container(height=300)

if __name__ == "__main__":
    main()
