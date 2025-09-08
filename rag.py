import os
import time
import logging
from typing import List, Tuple, Optional

from langchain_perplexity import ChatPerplexity
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class RAGPipeline:
    def __init__(
        self,
        pplx_api_key: str,
        hf_api_key: Optional[str] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        prefer_endpoint: bool = True,
        endpoint_probe_timeout: float = 8.0,
    ):
        self.pplx_api_key = pplx_api_key
        self.hf_api_key = hf_api_key
        self.persist_directory = persist_directory or settings.PERSIST_DIRECTORY
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.prefer_endpoint = prefer_endpoint
        self.endpoint_probe_timeout = endpoint_probe_timeout

        self.llm = ChatPerplexity(
            model=settings.MODEL_NAME,
            temperature=settings.MODEL_TEMPERATURE,
            pplx_api_key=pplx_api_key,
        )
        self.embeddings = None
        self.vs: Optional[Chroma] = None
        self.retriever = None
        self.chain = None

    def _get_embeddings(self):
        # firts HF Endpoint, if not OK -> locally
        if self.prefer_endpoint and self.hf_api_key:
            try:
                logging.info("Trying HuggingFace Endpoint embeddings...")
                emb = HuggingFaceEndpointEmbeddings(
                    model=settings.EMBEDDING_MODEL_NAME,
                    huggingfacehub_api_token=self.hf_api_key,
                )
                t0 = time.time()
                emb.embed_query("ping")  # test call
                if time.time() - t0 > self.endpoint_probe_timeout:
                    raise TimeoutError("HF endpoint probe took too long")
                return emb
            except Exception as e:
                logging.warning(f"HF Endpoint unavailable/slow, fallback to local: {e}")

        logging.info("Loading local HuggingFace embeddings ...")
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )

    def index_pdf(self, pdf_path: str) -> None:
        # Load, stplit, build vector-store retriever and chain
        loader = UnstructuredLoader(
            file_path=pdf_path,
            languages=["rus", "eng"],
        )
        documents = list(loader.lazy_load())
        logging.info(f"Loaded {len(documents)} raw document(s)")

        texr_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = texr_splitter.split_documents(documents)
        chunks= filter_complex_metadata(chunks)
        logging.info(f"Split into {len(chunks)} chunk(s)")
        

        if self.embeddings is None:
            self.embeddings = self._get_embeddings()
            logging.info("Got embedder")

        is_existent = os.path.isdir(self.persist_directory) and os.listdir(self.persist_directory)
        if is_existent:
            logging.info("Loading existing Chroma collection from disk...")
            self.vs = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            logging.info("Loaded Chroma collection from disk...")
        if not chunks:
            raise ValueError("There is not document for creating Chroma")

        logging.info("Building new Chroma collection and persisting to disk...")
        self.vs = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        try:
            self.vs.persist()
        except Exception:
            pass
        
        logging.info("Loaded chroma")
        
        self.retriever = self.vs.as_retriever(search_kwargs={"k": 5})
        logging.info("Chroma as a retriever")
        
        
        # system_prompt = (
        #     "Ты ИИ ассистент, который ответит на вопросы по документу.\n"
        #     "Тебе будут даны части документа и вопрос. Предоставь развернутый ответ.\n"
        #     "Если ответа нет в контексте или документе, то так и скажи.\n\n"
        #     "Не добавляй в ответ сноски вида [1], [2], [3-5] и любые квадратные скобки с цифрами.\n\n"
        #     "Не добавляй ** в ответе.\n\n"
        #     "Контекст:\n{context}"
        #     "=================\n"
        #     "Ответ:\n"
        # )
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", system_prompt),
        #         ("human", "{input}"),
        #     ]
        # )
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Ты ИИ ассистент, который ответит на вопросы по документу.\n"
                "Не добавляй в ответ сноски/цитаты вида [1], [2], [3-5] и любые квадратные скобки с цифрами.\n"
                "Также не добавляй ** в ответ.\n\n"
                "Тебе нельзя выдумывать ответы, отвечай строго по документу.\n\n"
                "Если пользователь пишет что-то непонятное или просто вкидывает разные словечки (не по тексту),\n\n"
                "То, что вообще в тексте нет или не может быть, то скажи пользователю задавать вопросы по файлу.\n\n"
                "Ничего другого ему не пиши, не пытайся как-то ответить на что-то непонятное, что он написал.\n\n"
                "Если ты не нашёл совпадений в контексте или документе, то так и скажи.\n\n"
                "Контекст:\n{context}\n\n"
                "Вопрос: {question}\n"
                "Ответ:"
            ),
        )
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        # doc_chain = create_stuff_documents_chain(self.llm, prompt)
        # self.chain = create_retrieval_chain(self.retriever, doc_chain)
        logging.info("RAG chain was made")

    def ask(self, query: str) -> Tuple[str, List[Document]]:
        if not self.chain:
            raise RuntimeError("RAG is not initialized, call index_pdf().")
        
        docs = self.retriever.invoke(query)
        result = self.chain.invoke({"query": query, "input_documents": docs})
        
        # res = self.chain.invoke({"input": query})
        logging.info("RAG chain invoked")
        # return res.get("answer", "").strip(), res.get("context", [])
        return result["result"]

    # @staticmethod
    # def format_sources(docs: List[Document]) -> List[str]:
    #     out = []
    #     for d in docs:
    #         src = d.metadata.get("source") or d.metadata.get("filename") or "unknown"
    #         page = d.metadata.get("page_number", d.metadata.get("page"))
    #         if page is not None:
    #             out.append(f"{src} (page {page})")
    #         else:
    #             out.append(src)
    #     return out
