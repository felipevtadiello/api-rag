import os
import tempfile
import re
import fitz 
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Form
from typing import Annotated, List
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.chains import ConversationalRetrievalChain
import pinecone

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
API_KEY = os.getenv("API_BACKEND_KEY")
API_KEY_NAME = "X-Api-Key"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.5)
app = FastAPI(title="API do Chat RAG Robusta")

api_key_header_scheme = Header(alias=API_KEY_NAME)
async def api_key_auth(x_api_key: Annotated[str, api_key_header_scheme]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Chave de API inválida ou ausente")

def process_pdf_text(pdf_path):
    full_text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                full_text += page.get_text("text") + "\n\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler o arquivo PDF: {e}")
    
    full_text = re.sub(r' +', ' ', full_text)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    return full_text

def create_text_chunks(full_text):
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(full_text)


@app.get("/list-documents/", response_model=List[str], dependencies=[Depends(api_key_auth)])
def list_documents():
    try:
        query_response = pinecone_index.query(
            vector=[0.0] * 768,
            top_k=10000,
            include_metadata=True
        )
        sources = set()
        for match in query_response['matches']:
            if 'source' in match['metadata']:
                sources.add(match['metadata']['source'])
        return sorted(list(sources))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao listar documentos do Pinecone: {str(e)}")

@app.post("/upload-and-process/", dependencies=[Depends(api_key_auth)])
async def upload_and_process(file: UploadFile = File(...), doc_name: str = Form(...)):
    
    allowed_content_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain", "text/markdown"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(status_code=400, detail=f"Tipo de arquivo '{file.content_type}' não suportado.")

    existing_docs = list_documents()
    if doc_name in existing_docs:
        raise HTTPException(status_code=400, detail=f"Um documento com o nome '{doc_name}' já foi processado.")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        raw_text = process_pdf_text(tmp_path)
        chunks = create_text_chunks(raw_text)
        
        metadatas = [{"source": doc_name} for _ in chunks]
        LangchainPinecone.from_texts(texts=chunks, embedding=embeddings_model, metadatas=metadatas, index_name=PINECONE_INDEX_NAME)
        
        return {"status": "sucesso", "filename": doc_name, "message": f"Documento '{doc_name}' processado e adicionado."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro: {str(e)}")
    finally:
        os.remove(tmp_path)

class DocumentRequest(BaseModel):
    filename: str

@app.post("/delete-document/", dependencies=[Depends(api_key_auth)])
def delete_document(request: DocumentRequest):
    try:
        filename_to_delete = request.filename
        pinecone_index.delete(filter={"source": {"$eq": filename_to_delete}})
        return {"status": "sucesso", "filename": filename_to_delete, "message": "Documento removido do Pinecone."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao remover o documento: {str(e)}")

class QuestionRequest(BaseModel):
    question: str
    chat_history: list = []

@app.post("/ask/", dependencies=[Depends(api_key_auth)])
def ask_question(request: QuestionRequest):
    try:
        vectorstore = LangchainPinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings_model)
        
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
            return_source_documents=True
        )
        
        response = conversation_chain({'question': request.question, 'chat_history': formatted_history})
        
        source_documents_formatted = []
        if 'source_documents' in response:
            for doc in response['source_documents']:
                source_documents_formatted.append({
                    "page_content": doc.page_content,
                    "source": doc.metadata.get('source', 'N/A')
                })
        
        return {
            "answer": response['answer'], 
            "source_documents": source_documents_formatted
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao processar a pergunta: {str(e)}")