import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)

app = FastAPI(title="API do Chat RAG")

def processar_texto_de_pdf(pdf_file):
    texto = ""
    try:
        leitor_pdf = PdfReader(pdf_file)
        for pagina in leitor_pdf.pages:
            texto_pagina = pagina.extract_text()
            if texto_pagina:
                texto += texto_pagina
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler PDF: {e}")
    return texto

def criar_chunks(texto):
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(texto)

@app.post("/upload-and-process/")
async def upload_and_process(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Tipo de arquivo inválido. Por favor, envie um PDF.")
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        raw_text = processar_texto_de_pdf(tmp_path)
        chunks = criar_chunks(raw_text)
        Pinecone.from_texts(texts=chunks, embedding=embeddings, index_name=PINECONE_INDEX_NAME)
        os.remove(tmp_path)
        return {"status": "sucesso", "filename": file.filename, "message": "Documento processado e adicionado."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro: {str(e)}")

class QuestionRequest(BaseModel):
    question: str
    chat_history: list = []

@app.post("/ask/")
def ask_question(request: QuestionRequest):
    try:
        vectorstore = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
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
        raise HTTPException(status_code=500, detail=f"Erro ao processar a pergunta: {str(e)}")