import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import docx2txt
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from pydantic import BaseModel




# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize Pinecone client
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
port = os.getenv('PORT')

# Define Pinecone client instance
pc = Pinecone(api_key=pinecone_api_key)

# Define embedding function and dimension
embedding_model_name = "all-MiniLM-L6-v2"
embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
sample_embedding = embedding_function.embed_query("test")
embedding_dim = len(sample_embedding)
conversational_memory_length = 10  # number of previous messages the chatbot will remember during the conversation

memory = ConversationBufferWindowMemory(
    k=conversational_memory_length, 
    memory_key="chat_history", 
    return_messages=True
)

# Initialize vector store
def initialize_vector_store(index_name):
    """Create and return a Pinecone vector store for the given index name."""
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return PineconeVectorStore(index_name=index_name, embedding=embedding_function)

# Pydantic model for question requests
class QuestionRequest(BaseModel):
    question: str


# Document upload endpoint with namespaces for each candidate
@app.post("/upload-document/{index_name}/{document_id}")
async def upload_document(index_name: str, document_id: str, file: UploadFile = File(...)):
    """Endpoint to upload and process a document into a specific story's index under a story's namespace."""
    try:
        # Initialize vector store for the company
        vector_store = initialize_vector_store(index_name)

        # Read and process the document
        if file.content_type == "application/pdf":
            pdf_reader = PdfReader(file.file)
            text = "".join([page.extract_text() for page in pdf_reader.pages])
        elif file.content_type == "text/plain":
            text = (await file.read()).decode("utf-8")
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = docx2txt.process(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Split text into chunks
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        text_chunks = text_splitter.split_text(text)
        
        # Add text chunks to Pinecone under the candidate's namespace
        vector_store.add_texts(text_chunks, namespace=document_id)
        
        return JSONResponse(status_code=200, content={"message": "Document processed and uploaded successfully"})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


# Query endpoint for a candidate's namespace
@app.post("/query-doc/{index_name}/{document_id}")
async def ask_question(index_name: str, document_id: str, request: QuestionRequest):
    """Endpoint to query a specific candidate's document in the specified company's Pinecone index."""
    try:
        # Initialize vector store for the company
        vector_store = initialize_vector_store(index_name)

        # Set up conversational chain
        llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama-3.2-90b-vision-preview')
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type='stuff',
            retriever=vector_store.as_retriever(search_kwargs={"k": 2, "namespace": document_id}),
            memory=memory
        )

        # Get response for the query
        response = chain.invoke({"question": request.question, "chat_history": []})
        return JSONResponse(status_code=200, content={"answer": response["answer"]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_context():
    """Endpoint to reset the chat context (conversation memory)."""
    try:
        # Clear the conversation memory
        memory.clear()
        return JSONResponse(status_code=200, content={"message": "Chat context has been reset."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(port))




