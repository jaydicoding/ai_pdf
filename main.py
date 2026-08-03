from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st
import tempfile
import os

load_dotenv()

#Title
st.title("AI PDF")
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load()
    return pages

#File upload
uploaded_file = st.file_uploader("Chooses PDF files")
st.write("---")

#Code that runs when uploaded
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 100,
        length_function = len,
        is_separator_regex = False,
    )

    texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings()

    #Load it into Chroma
    db= Chroma.from_documents(
        documents=texts, 
        embedding=embeddings_model,
    )

    #Question
    st.header("Ask questions about the PDF")
    question = st.text_input("Enter your question")

    if st.button("Ask"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            with st.spinner("Searching for an answer..."):

                llm = ChatOpenAI(
                    model="gpt-4.1-mini",
                    temperature=0,
                )

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """
                            Answer the user's question using only the
                            following PDF context.

                            If the answer cannot be found in the context,
                            say that the answer cannot be found in the PDF.

                            Context:
                            {context}
                            """,
                        ),
                        (
                            "human",
                            "{input}",
                        ),
                    ]
                )

                document_chain = create_stuff_documents_chain(
                    llm,
                    prompt,
                )

                retriever = db.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": min(8, len(texts)),
                        "fetch_k": min(20, len(texts)),
                    },
                )

                retrieval_chain = create_retrieval_chain(
                    retriever,
                    document_chain,
                )

                result = retrieval_chain.invoke(
                    {
                        "input": question,
                    }
                )

                st.subheader("Answer")
                st.write(result["answer"])




