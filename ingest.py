"""Módulo de Ingesta (Setup).

Carga los documentos de /data, los fragmenta (chunking) y los persiste en una
colección de ChromaDB. Si el índice ya existe en disco, lo reutiliza en vez de
reindexar — evita gastar tiempo y cómputo cada vez que se corre el pipeline.

También se puede correr como script suelto: `python ingest.py`.
"""

import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"
PERSIST_DIR = "./vectorstore"
COLLECTION_NAME = "rag_demo"

# Mismo modelo de embeddings acá (para indexar) y en rag.py (para consultar).
# Si no coincidieran, los vectores vivirían en espacios distintos y la búsqueda
# por similitud dejaría de tener sentido (el error #1 más común en RAG).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def cargar_documentos(data_dir: str = DATA_DIR):
    """Carga todos los .txt y .md de data_dir."""
    documentos = []
    for patron in ("*.txt", "*.md"):
        loader = DirectoryLoader(
            data_dir, glob=patron, loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        documentos.extend(loader.load())
    return documentos


def fragmentar(documentos, chunk_size: int = 500, chunk_overlap: int = 70):
    """Chunking en TOKENS, no en caracteres.

    El límite de contexto del LLM se mide en tokens: un chunk de "500 caracteres"
    puede equivaler a una cantidad de tokens muy distinta según el idioma y la
    puntuación. from_tiktoken_encoder hace que chunk_size=500 sea literalmente
    500 tokens. Cumple el mínimo pedido (500 tokens, 50 de overlap); se usa un
    overlap de 70 como margen.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documentos)


def ingestar(
    data_dir: str = DATA_DIR,
    persist_dir: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Punto de entrada del módulo de ingesta.

    Si ya existe un índice persistido en persist_dir, lo carga sin reindexar.
    Si no existe, carga los documentos de data_dir, los fragmenta y los indexa.
    """
    embeddings = get_embeddings()
    ya_existe_indice = os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0

    if ya_existe_indice:
        print(f"♻️  Índice existente en {persist_dir} — se reutiliza sin reindexar")
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    print(f"🆕 No hay índice previo en {persist_dir} — indexando documentos de /{data_dir}")
    documentos = cargar_documentos(data_dir)
    chunks = fragmentar(documentos)
    print(f"📄 Documentos cargados: {len(documentos)} · ✂️ Fragmentos generados: {len(chunks)}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    print(f"📦 Índice creado con {vectorstore._collection.count()} fragmentos")
    return vectorstore


if __name__ == "__main__":
    ingestar()
