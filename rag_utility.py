import os
from typing import Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    try:
        from langchain_classic.chains import RetrievalQA
    except ImportError:
        RetrievalQA = Any

Document = Any

# Load environment variables from .env file
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the embedding model
embedding = HuggingFaceEmbeddings()

# Load the Llama-3.3-70B model from Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

VECTORSTORE_DIR = os.path.join(working_dir, "doc_vectorstore")


def process_documents_to_chroma_db(file_names: list[str]) -> int:
    """
    Process multiple PDF documents and store their chunks in ChromaDB.
    Each chunk is tagged with its source file name as metadata.

    Args:
        file_names: List of PDF file names (already saved to working_dir)

    Returns:
        Total number of chunks stored
    """
    all_texts: list[Document] = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    for file_name in file_names:
        file_path = os.path.join(working_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping.")
            continue

        # Load PDF
        loader = UnstructuredPDFLoader(file_path)
        documents = loader.load()

        # Tag each document chunk with the source file name
        for doc in documents:
            doc.metadata["source_file"] = file_name

        # Split into chunks
        chunks = text_splitter.split_documents(documents)

        # Ensure source_file propagates to all split chunks
        for chunk in chunks:
            chunk.metadata.setdefault("source_file", file_name)

        all_texts.extend(chunks)

    if not all_texts:
        raise ValueError("No documents were loaded. Please check the uploaded files.")

    # Build / rebuild the Chroma vector store with all chunks
    vectordb = Chroma.from_documents(
        documents=all_texts,
        embedding=embedding,
        persist_directory=VECTORSTORE_DIR
    )

    return len(all_texts)


def answer_question(user_question: str) -> tuple[str, list[str]]:
    """
    Answer a question using RAG over the stored vector database.

    Args:
        user_question: The user's question string.

    Returns:
        A tuple of (answer_text, list_of_source_file_names)
    """
    # Load the persisted Chroma vector database
    vectordb = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embedding
    )

    # Retriever: fetch top-k relevant chunks
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # Retrieve relevant documents first so we can extract sources
    relevant_docs = retriever.invoke(user_question)

    # Collect unique source file names from retrieved chunks
    sources = list(
        dict.fromkeys(
            doc.metadata.get("source_file", "Unknown")
            for doc in relevant_docs
        )
    )

    # Build the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer, sources
