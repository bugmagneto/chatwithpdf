import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import faiss
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain.llms import 

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function= len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = faiss.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
   # llm = 
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        #llm = llm,
        retriever= vectorstore.as_retriever,
        memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs",page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with Multiple PDFs :books:")
    st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs= st.file_uploader("Upload your PDFs here and Click on'Process'",accept_multiple_files=True)
        if  st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)


        

if __name__ == '__main__':
    main()