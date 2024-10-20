import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import docx2txt
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
import json

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

# Memory for chat history (temporary in-memory storage)
conversation_memories = {}
system_prompt = 'You are a friendly conversational chatbot'
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{human_input}"),
])

conversational_memory_length = 5  # number of previous messages the chatbot will remember during the conversation

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

# Pydantic model for chat messages
# Define the request model
class ChatRequest(BaseModel):
    user_message: str

@app.post("/upload-document/{index_name}")
async def upload_document(index_name: str, file: UploadFile = File(...)):
    """Endpoint to upload and process a document into the specified Pinecone index."""
    try:
        # Initialize vector store for the index
        vector_store = initialize_vector_store(index_name)

        # Read the document
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

        # Add text chunks to Pinecone
        vector_store.add_texts(text_chunks)

        return JSONResponse(status_code=200, content={"message": "Document processed and uploaded successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-doc/{index_name}")
async def ask_question(index_name: str, request: QuestionRequest):
    """Endpoint to query the document content in the specified Pinecone index."""
    try:
        # Initialize vector store for the index
        vector_store = initialize_vector_store(index_name)

        # Create conversational retrieval chain
        llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-70b-8192')
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type='stuff',
            retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
            memory=memory
        )

        # Query Pinecone and get response
        response = chain.invoke({"question": request.question, "chat_history": []})
        return JSONResponse(status_code=200, content={"answer": response["answer"]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    # Define schema for structured output
    class ResumeSchema(BaseModel):
        name: str = Field(alias="Name")
        address: str = Field(alias="Address")
        linkedin: str = Field(alias="LinkedIn URL")
        phone: str = Field(alias="Phone number")
    """Endpoint to extract data from a resume."""
    try:
        # Extract text from the uploaded document
        text = ""
        if file.content_type == "application/pdf":
            try:
                pdf_reader = PdfReader(file.file)
                text = "".join([page.extract_text() for page in pdf_reader.pages])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading PDF file: {str(e)}")
        elif file.content_type == "text/plain":
            try:
                text = (await file.read()).decode("utf-8")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading text file: {str(e)}")
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                text = docx2txt.process(file.file)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading DOCX file: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Log the extracted text for diagnostics
        print(f"Extracted Text: {text[:500]}...")  # Logging first 500 characters

        # Initialize Groq AI with structured output schema
        llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-70b-8192')
        structured_llm = llm.with_structured_output(schema=ResumeSchema)

        # Define prompt for extracting resume information
        prompt = (
                f"Please extract the following details from the Resume text in JSON format, response with 'none' if empty: "
                f"1. Name, 2. Address, 3. LinkedIn URL, 4. Phone number. "
                f"Here is the resume text: {text}"
            )
        # Invoke the structured output model
        try:
            response = structured_llm.invoke(prompt)
            print(f"Structured JSON Response: {response}")  # Log response for debugging
        except Exception as e:
            print(f"Error parsing Groq AI response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error parsing Groq AI response: {str(e)}")

        return response

    except Exception as e:
        print(f"Unhandled Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    """Chat with AI, retaining chat history."""
    try:
        # Initialize Groq model
        llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-70b-8192')

        system_prompt = 'You are a friendly conversational chatbot'


        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),  # System message
                MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history
                HumanMessagePromptTemplate.from_template("{human_input}"),  # User input placeholder
            ]
        )

        # Create a conversation chain using the LangChain LLM
        conversation = LLMChain(
            llm=llm,  # Groq LangChain chat object initialized earlier
            prompt=prompt,  # The constructed prompt template
            verbose=False,  # Disable verbose output for debugging
            memory=memory,  # The conversational memory object that stores and manages the conversation history
        )

        # The chatbot's answer is generated by sending the full prompt to the Groq API
        response = conversation.predict(human_input=request.user_message)

        # Return the response from the chatbot
        return JSONResponse(status_code=200, content={"answer": response})

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




