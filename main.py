import os
import boto3
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

aws_access_key_id=os.getenv("aws_access_key_id")
aws_secret_access_key=os.getenv("aws_secret_access_key")
#region_name=os.getenv("region_name")

# Setting up bedrock client



bedrock=boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

bedrock_embedding=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with 
250 words with detailed explantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

def get_documents():
    loader = PyPDFDirectoryLoader('data')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    docs=text_splitter.split_documents(documents)
    return docs


def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(docs,bedrock_embedding)
    vectorstore_faiss.save_local('faiss_local_vectordb')
    
def get_llm():
    llm=Bedrock(model_id="mistral.mistral-7b-instruct-v0:2",client=bedrock)
    return llm

PROMPT=PromptTemplate(template=prompt_template, input_variables=['context','question'])

def get_llm_response(llm,vectorstore_faiss,query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type='similarity',search_kwargs={'k':3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents = True
       
    )
    response=qa({'query':query})
    return response['result']

def main():
    st.set_page_config('RAG')
    st.header('End to End RAG using Amazon Bedrock')
    user_question=st.text_input('Ask a question from PDF')

    with st.sidebar:
        st.subheader('Choose a PDF')
        pdf_list=os.listdir('data')
        pdf_list.sort()
        selected_pdf=st.selectbox('Select a PDF',pdf_list)
        st.write(f'You have selected {selected_pdf}')

        st.title('Update and create vector store')

        if st.button('Store Vector'):
            docs=get_documents()
            get_vector_store(docs)
            st.success('Done! Vector store created')

    if st.button('Get Answer'):
        llm=get_llm()
        vectorstore_faiss=FAISS.load_local('faiss_local_vectordb',bedrock_embedding, allow_dangerous_deserialization=True)
        response=get_llm_response(llm,vectorstore_faiss,user_question)
        st.write(response)

if __name__ == '__main__':
    main()




