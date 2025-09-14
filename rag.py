import os
import time
import logging
from typing import Optional, Literal
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores.utils import filter_complex_metadata


from langchain_perplexity import ChatPerplexity

from config import settings
from prompt import PROMPT_RU, PROMPT_ENG

logger = logging.getLogger("rag_bot")
logger.setLevel(logging.INFO)

def select_llm(api_key: str):
    return ChatPerplexity(
        model=settings.MODEL_NAME,
        temperature=settings.MODEL_TEMPERATURE,
        pplx_api_key=api_key,
    )
def select_embeddings(prefer_endpoint: bool, hf_api_key: Optional[str] = None):
    if prefer_endpoint and hf_api_key:
        try:
            logger.info("Пробуем HuggingFace Endpoint Embeddings - API")
            emb = HuggingFaceEndpointEmbeddings(
                model=settings.EMBEDDING_MODEL_NAME,
                huggingfacehub_api_token=hf_api_key,
            )
            t0 = time.time()
            emb.embed_query("ping")
            if time.time() - t0 > 8:
                raise TimeoutError("HF endpoint медленно")
            return emb
        except Exception as e:
            logger.warning(f"HF endpoint не доступен, fallback to local: {e}")
    logger.info("Локальные HuggingFaceEmbeddings")
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

def ensure_chroma(docs, persist_directory, collection_name, embeddings):
    if os.path.isdir(persist_directory) and any(os.scandir(persist_directory)):
        logger.info("Загружаем Chroma с диска...")
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
    logger.info("Создаём новый Chroma store...")
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return store

def build_qa_chain(
    pdf_path: str,
    api_key: str,
    hf_api_key: Optional[str] = None,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    prefer_endpoint: bool = True,
    language: Literal["ru", "eng"] = "ru",
) -> RetrievalQA:
    persist_directory = persist_directory or settings.PERSIST_DIRECTORY
    collection_name = collection_name or settings.COLLECTION_NAME
    prompt = PROMPT_RU if language == "ru" else PROMPT_ENG
    
    loader = UnstructuredLoader(
        file_path=pdf_path,
        languages=["rus", "eng"] if language == "ru" else ["eng"],
    )
    documents = list(loader.lazy_load())
    logger.info(f"Загружено {len(documents)} сегментов PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    chunks= filter_complex_metadata(chunks)
    logger.info(f"Чанковано: {len(chunks)}")

    embeddings = select_embeddings(prefer_endpoint, hf_api_key)
    vectordb = ensure_chroma(chunks, persist_directory, collection_name, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    llm = select_llm(api_key)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def ask(qa_chain: RetrievalQA, question: str) -> str:
    """Ask the question and return the answer."""
    logger.info(f"Запрос: {question}")
    result = qa_chain.invoke({"query": question})
    answer = result.get("result", "")
    return answer
