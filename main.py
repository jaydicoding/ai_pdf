from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st
import tempfile
import os


load_dotenv()

st.set_page_config(
    page_title="AI PDF",
    page_icon="📄",
)

st.title("AI PDF")
st.caption("Upload a PDF and ask questions about its content.")
st.write("---")


@st.cache_resource
def create_vector_db(file_bytes):
    """Create a vector database from an uploaded PDF."""

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:
        temp_file.write(file_bytes)
        temp_filepath = temp_file.name

    try:
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )

        texts = text_splitter.split_documents(pages)

        if not texts:
            raise ValueError(
                "No readable text was found in the PDF."
            )

        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings_model,
        )

        return db, len(pages), len(texts)

    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
)

st.write("---")


if uploaded_file is not None:
    try:
        with st.spinner("Processing the PDF..."):
            db, page_count, chunk_count = create_vector_db(
                uploaded_file.getvalue()
            )

        st.success(
            f"PDF processed successfully: "
            f"{page_count} pages, {chunk_count} chunks"
        )

        st.header("Ask questions about the PDF")

        question = st.text_input(
            "Enter your question",
            placeholder="What is this document about?",
        )

        if st.button("Ask", type="primary"):
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
                                Answer the user's question using only
                                the supplied PDF context.

                                Answer in the same language as the
                                user's question.

                                If the answer cannot be found in the
                                context, clearly say that it cannot
                                be found in the PDF.

                                PDF context:
                                {context}
                                """,
                            ),
                            (
                                "human",
                                "{input}",
                            ),
                        ]
                    )

                    document_chain = (
                        create_stuff_documents_chain(
                            llm,
                            prompt,
                        )
                    )

                    retriever = db.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": min(8, chunk_count),
                            "fetch_k": min(20, chunk_count),
                        },
                    )

                    retrieval_chain = (
                        create_retrieval_chain(
                            retriever,
                            document_chain,
                        )
                    )

                    result = retrieval_chain.invoke(
                        {
                            "input": question,
                        }
                    )

                st.subheader("Answer")
                st.write(result["answer"])

                source_pages = sorted(
                    {
                        doc.metadata.get("page", 0) + 1
                        for doc in result.get("context", [])
                    }
                )

                if source_pages:
                    st.caption(
                        "Sources: Page "
                        + ", ".join(
                            map(str, source_pages)
                        )
                    )

    except Exception as error:
        st.error(f"An error occurred: {error}")

else:
    st.info("Upload a PDF file to begin.")




