
import streamlit as st

from zhipuai_llm import ZhipuaiLLM

import os 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch,RunnablePassthrough

import sys 
from langchain_chroma import Chroma
from zhipuai_embedding import ZhipuAIEmbeddings


sys.path.append("notebook/C3 搭建知识库")

zhipu_api_key = os.environ['ZHIPUAI_API_KEY']

#初始化chroma 向量库检索器
def get_retriever():
    embedding = ZhipuAIEmbeddings()
    persist_directory = 'data_base/vector_db/chroma'
    vectordb = Chroma(persist_directory=persist_directory,embedding_function=embedding)
    return vectordb.as_retriever()

#chroma 向量库检索文本内容
def combine_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs['context'])

def get_qa_history_chain():

    #调用向量库的retriever
    retriever = get_retriever()

    #大模型
    llm = ZhipuaiLLM(api_key=zhipu_api_key, model_name='GLM-5.3-FLASH', temperature=0.1)

    #系统提示词
    condense_question_system_template = ('请根据聊天记录总结用户最近的问题,'
                                         '如果没有多余的聊天记录则返回用户的问题')

    #消息提示模板
    condense_quest_prompt = ChatPromptTemplate([
        ('system', condense_question_system_template),
        ('placeholder','{chat_history}'),
        ('human','{input}')
    ])

    #使用消息分支执行 必须是runnable对象
    retrieve_docs = RunnableBranch(
        (lambda x: not x.get('chat_history',False),(lambda x: x['input'])|retriever),
        condense_quest_prompt | llm | StrOutputParser() | retriever
        )

    #系统prompt
    system_propmt = (
        '你是一个问答任务的助手。'
        '请使用检索到的上下文判断回答这个问题。'
        '如果你不知道就说不知道。'
        '请使用简介的话语回答用户。'
        '\n\n'
        '{context}'
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_propmt),
            ('placeholder','{chat_history}'),
            ('human','{input}')
         ]
    )

    qa_chian = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chian = RunnablePassthrough().assign(
        context = retrieve_docs,
    ).assign(answer=qa_chian)

    return qa_history_chian

def gen_response(chain, input, chat_history):
    response = chain.stream({
        'input': input,
        'chat_history': chat_history
    })

    for res in response:
        if 'answer' in res.key():
            yield res['answer']

def main():
    st.markdown = '动手学大模型应用开发'

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'qa_history_chain' not in st.session_state:
        st.session_state.qa_history_chian = get_qa_history_chain()

    #建立容器高度为500px
    messages = st.container(height=550)
    #显示整个对话历史
    for message in st.session_state.messages: #遍历对话历史
        with messages.chat_message(message[0]): #messages 只在容器下显示，chat_message显示用户以及ai头像
            st.write(message[1]) #打印内容

    if prompt := st.chat_input('Say something'):
        #将用户输入添加到历史对话中
        st.session_state.messages.append(('human',prompt))
        #显示当前用户输入
        with messages.chat_message('human'):
            st.write(prompt)

        #生成回复
        answer = gen_response(chain = st.session_state.qa_history_chian,
                              input = prompt,
                              chat_history = st.session_state.messages)

        #流式输出
        with messages.chat_message('ai'):
            output = st.write_stream(answer)

        st.session_state.messages.append(('ai', output))


