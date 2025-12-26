import boto3
import json
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.llms.bedrock import Bedrock
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

bedrock=boto3.client(service_name='bedrock-runtime',region_name=AWS_REGION)
embeddings=BedrockEmbeddings(model_id='amazon.titan-embed-text-v1',client=bedrock)

# Data Ingestion and Text splitting
@st.cache_resource
def data_ingestion():
    loader=PyPDFDirectoryLoader("research_papers")
    documents=loader.load()
    text_splitters=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=250)
    docs=text_splitters.split_documents(documents=documents)
    return docs

# Create embeddings and store it in vector store
@st.cache_resource
def vectorstore_retriever(docs):

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store=FAISS.from_documents(docs,embeddings)
        st.session_state.retriever=st.session_state.vector_store.as_retriever(
            search_type='similarity',search_kwargs={'k':3}
        )

prompt_template="""
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast 
1500 words to summarize with detailed explantions. If you don't know the answer, 
just say that you don't know, don't try to hallucinate the answer.
<context>
{context}
</context>

Question: {question}

Assistant:
 """

def get_llama3_llm(retriever,query):
    llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock,
                model_kwargs={'max_gen_len':3000})
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(query)

    return response


def get_amazon_llm(query,context):

    model_id="amazon.nova-lite-v1:0"
    user_message_text = f"""
    Use the following pieces of context to provide a detailed answer to the question below.
    Use at least 1500 words to explain thoroughly with examples if possible.
    If you don't know the answer, just say that you don't know; do not hallucinate.

    <context>
    {context}
    </context>

    Question:
    {query}
    """
    body = json.dumps({
        "messages": [
            {"role": "user", "content": [{"text": user_message_text}]}
        ],
        "inferenceConfig": {
            "maxTokens": 4000,   # Increase maxTokens to allow longer answers
            "temperature": 0.7,
            "topP": 0.9
        }
    })
    accept = "application/json"
    content_type = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )
    response_body = json.loads(response.get("body").read())

    return response_body["output"]["message"]["content"][0]["text"]





prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])



def main():

    st.set_page_config('Research papers Q and A')
    st.header("Q and A with AWS BedRock for Research Papers")
    st.subheader("Research Papers:\n 1. Attention All You Need \n 2. A Comprehensive Overview of Large Language Models")
    user_question=st.text_input('Ask any question from the Research papers')

    with st.sidebar:
        st.title('create vector store')

        if st.button('vectors creation'):
            with st.spinner('creating vector embeddings and storing in FAISS DB...'):
                docs=data_ingestion()
                vectorstore_retriever(docs)
                st.success('vector store created')

    if st.button('Llama Response'):
        if 'vector_store' not in st.session_state:
            st.warning("Please create vector store first from sidebar.")
        else:
            with st.spinner('Retrieving Response...'):
                
                
                st.write(get_llama3_llm(st.session_state.retriever,user_question))
                st.success('Done')

    if st.button('Amazon Response'):
        if 'vector_store' not in st.session_state:
            st.warning("Please create vector store first from sidebar.")
        else:
            with st.spinner('Retrieving Response...'):
                
                similar_docs = st.session_state.vector_store.similarity_search(user_question)
                context = "\n\n".join([doc.page_content for doc in similar_docs])
                st.write(get_amazon_llm(user_question,context))
                st.success('Done')

if __name__=='__main__':
    main()