import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from typing import Annotated
from pydantic import BaseModel
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

API_KEY = os.getenv("API_BACKEND_KEY")
API_KEY_NAME = "X-Api-Key"

api_key_header_scheme = Header(alias=API_KEY_NAME)


async def api_key_auth(x_api_key: Annotated[str, api_key_header_scheme]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Chave de API inválida ou ausente")


embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)

app = FastAPI(title="RAG Chat API")


def process_pdf_text(pdf_file):
    full_text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler o arquivo PDF: {e}")
    return full_text


def create_text_chunks(full_text):
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(full_text)


@app.post("/upload-and-process/", dependencies=[Depends(api_key_auth)])
async def upload_and_process(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Tipo de arquivo inválido. Por favor, envie um PDF.")
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        raw_text = process_pdf_text(tmp_path)
        chunks = create_text_chunks(raw_text)
        Pinecone.from_texts(texts=chunks, embedding=embeddings_model, index_name=PINECONE_INDEX_NAME)
        os.remove(tmp_path)

        return {"status": "sucesso", "filename": file.filename,
                "message": "Documento processado e adicionado à base de conhecimento."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro inesperado: {str(e)}")


class QuestionRequest(BaseModel):
    question: str
    chat_history: list = []


@app.post("/ask/", dependencies=[Depends(api_key_auth)])
def ask_question(request: QuestionRequest):
    try:
        vectorstore = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings_model)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True
        )
        response = conversation_chain({'question': request.question, 'chat_history': request.chat_history})
        return {"answer": response['answer']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao processar a pergunta: {str(e)}")