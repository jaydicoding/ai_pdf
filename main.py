from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Loader
loader = PyPDFLoader("unsu.pdf")
pages = loader.load()


#split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 20,
    length_function = len,
    is_separator_regex = False,
)

texts = text_splitter.split_documents(pages)
print("===== PDF Information =====")
print(f"Pages: {len(pages)}")
print(f"Chunks: {len(texts)}")
print("===========================")
print("\nFirst Chunk:\n")
print(texts[0].page_content)



