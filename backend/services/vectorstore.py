from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

def build_vector_store(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    return vector_store
