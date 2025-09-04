import os
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from typing import Annotated, List
from pydantic import BaseModel
from dotenv import load_dotenv

# Novas importações necessárias
from langchain.text_splitter import CharacterTextSplitter
from docling.document_converter import DocumentConverter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone

load_dotenv()

# --- CONFIGURAÇÕES E INICIALIZAÇÃO ---
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
API_KEY = os.getenv("API_BACKEND_KEY")
API_KEY_NAME = "X-Api-Key"
DOCS_LOG_FILE = "processed_docs.json"
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
app = FastAPI(title="API do Chat RAG Híbrida")

api_key_header_scheme = Header(alias=API_KEY_NAME)
async def api_key_auth(x_api_key: Annotated[str, api_key_header_scheme]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Chave de API inválida ou ausente")

def load_processed_docs() -> List[str]:
    if os.path.exists(DOCS_LOG_FILE):
        with open(DOCS_LOG_FILE, "r") as f:
            try: return json.load(f)
            except json.JSONDecodeError: return []
    return []

def save_processed_docs(docs_list: List[str]):
    with open(DOCS_LOG_FILE, "w") as f:
        json.dump(sorted(list(set(docs_list))), f, indent=4)

def create_text_chunks(full_text):
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(full_text)

# --- ENDPOINTS DA API ---

@app.post("/upload-and-process/", dependencies=[Depends(api_key_auth)])
async def upload_and_process(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Tipo de arquivo inválido.")
    processed_docs = load_processed_docs()
    if file.filename in processed_docs:
        raise HTTPException(status_code=400, detail=f"O arquivo '{file.filename}' já foi processado.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Lógica de processamento híbrida
        converter = DocumentConverter()
        docling_document = converter.convert(tmp_path).document
        raw_text = docling_document.export_to_markdown()

        chunks = create_text_chunks(raw_text)

        metadatas = [{"source": file.filename} for _ in chunks]
        Pinecone.from_texts(
            texts=chunks,
            embedding=embeddings_model,
            metadatas=metadatas,
            index_name=PINECONE_INDEX_NAME
        )

        processed_docs.append(file.filename)
        save_processed_docs(processed_docs)

        return {"status": "sucesso", "filename": file.filename, "message": "Documento processado com Docling e adicionado."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro durante o processamento: {str(e)}")
    finally:
        os.remove(tmp_path)

@app.get("/list-documents/", response_model=List[str], dependencies=[Depends(api_key_auth)])
def list_documents():
    return load_processed_docs()

class DocumentRequest(BaseModel):
    filename: str

@app.post("/delete-document/", dependencies=[Depends(api_key_auth)])
def delete_document(request: DocumentRequest):
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    try:
        filename_to_delete = request.filename
        processed_docs = load_processed_docs()
        if filename_to_delete not in processed_docs:
            raise HTTPException(status_code=404, detail="Documento não encontrado.")

        pinecone_index.delete(filter={"source": {"$eq": filename_to_delete}})

        processed_docs.remove(filename_to_delete)
        save_processed_docs(processed_docs)

        return {"status": "sucesso", "filename": filename_to_delete, "message": "Documento removido."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao remover o documento: {str(e)}")


class QuestionRequest(BaseModel):
    question: str
    chat_history: list = []

@app.post("/ask/", dependencies=[Depends(api_key_auth)])
def ask_question(request: QuestionRequest):
    try:
        vectorstore = Pinecone.from_existing_index(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings_model)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

        formatted_history = []
        for i in range(0, len(request.chat_history), 2):
            if i + 1 < len(request.chat_history):
                user_msg = request.chat_history[i]
                assistant_msg = request.chat_history[i+1]
                if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                    formatted_history.append((user_msg["content"], assistant_msg["content"]))

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True
        )
        response = conversation_chain({'question': request.question, 'chat_history': formatted_history})
        return {"answer": response['answer']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao processar a pergunta: {str(e)}")