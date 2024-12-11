import os
import logging
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import docx2txt
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
# Load environment variables
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Retrieve environment variables and validate them
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY must be set as an environment variable.")
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY must be set as an environment variable.")
port = os.getenv('PORT', '8000')
try:
    port = int(port)
except ValueError:
    raise ValueError("PORT environment variable must be an integer.")
# Create FastAPI instance
app = FastAPI()
# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
# Embedding configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
sample_embedding = embedding_function.embed_query("test")
embedding_dim = len(sample_embedding)
# Conversational memory configuration
CONVERSATIONAL_MEMORY_LENGTH = 10  # Number of previous messages to remember
memory = ConversationBufferWindowMemory(
    k=CONVERSATIONAL_MEMORY_LENGTH,
    memory_key="chat_history",
    return_messages=True
)
def initialize_vector_store(index_name: str) -> PineconeVectorStore:
    """Create and return a Pinecone vector store for the given index name."""
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        logger.info(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return PineconeVectorStore(index_name=index_name, embedding=embedding_function)
def read_document(file: UploadFile) -> str:
    """Read the uploaded file and return its text content."""
    if file.content_type == "application/pdf":
        pdf_reader = PdfReader(file.file)
        return "".join([page.extract_text() for page in pdf_reader.pages])
    elif file.content_type == "text/plain":
        return (file.file.read()).decode("utf-8")
    elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
def create_chain(vector_store: PineconeVectorStore, document_id: str) -> ConversationalRetrievalChain:
    """Create a conversational retrieval chain for querying documents."""
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama-3.3-70b-versatile')
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2, "namespace": document_id}),
        memory=memory
    )
    return chain
class QuestionRequest(BaseModel):
    question: str
@app.post("/upload-document/{index_name}/{document_id}")
async def upload_document(index_name: str, document_id: str, file: UploadFile = File(...)):
    """Endpoint to upload and process a document into a Pinecone index under a given namespace."""
    try:
        logger.info(f"Uploading document to index '{index_name}' under namespace '{document_id}'...")
        vector_store = initialize_vector_store(index_name)
        text = read_document(file)
        # Split text into chunks
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        text_chunks: List[str] = text_splitter.split_text(text)
        vector_store.add_texts(text_chunks, namespace=document_id)
        logger.info("Document processed and uploaded successfully.")
        return JSONResponse(status_code=200, content={"message": "Document processed and uploaded successfully"})
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/query-doc/{index_name}/{document_id}")
async def ask_question(index_name: str, document_id: str, request: QuestionRequest):
    """Query a document within a given Pinecone index namespace."""
    try:
        logger.info(f"Querying index '{index_name}', namespace '{document_id}' with question: {request.question}")
        vector_store = initialize_vector_store(index_name)
        chain = create_chain(vector_store, document_id)
        response = chain.invoke({"question": request.question, "chat_history": []})
        logger.info("Query processed successfully.")
        return JSONResponse(status_code=200, content={"answer": response["answer"]})
    except Exception as e:
        logger.error(f"Error querying document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/delete-namespace/{index_name}/{namespace}")
async def delete_namespace(index_name: str, namespace: str):
    """Delete a namespace from a given Pinecone index."""
    try:
        logger.info(f"Deleting namespace '{namespace}' from index '{index_name}'...")
        
        # Check if the index exists
        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' does not exist.")

        # Delete all vectors in the namespace
        index = pc.Index(index_name,"https://stories-uizc1dc.svc.aped-4627-b74a.pinecone.io")
        index.delete(delete_all=True, namespace=namespace)

        logger.info(f"Namespace '{namespace}' deleted successfully from index '{index_name}'.")
        return JSONResponse(status_code=200, content={"message": f"Namespace '{namespace}' deleted successfully."})
    except Exception as e:
        logger.error(f"Error deleting namespace: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_context():
    """Reset the chat context (conversation memory)."""
    try:
        memory.clear()
        logger.info("Chat context has been reset.")
        return JSONResponse(status_code=200, content={"message": "Chat context has been reset."})
    except Exception as e:
        logger.error(f"Error resetting chat context: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the server...")
    uvicorn.run(app, host="0.0.0.0", port=port)