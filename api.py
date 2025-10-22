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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.5)
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


@app.get("/list-courses/", response_model=List[str], dependencies=[Depends(api_key_auth)])
def list_courses():
    try:
        query_response = pinecone_index.query(
            vector=[0.0] * 768,
            top_k=10000,
            include_metadata=True
        )
        courses = set()
        for match in query_response['matches']:
            if 'course' in match['metadata']:
                courses.add(match['metadata']['course'])
        return sorted(list(courses))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao listar cursos: {str(e)}")


@app.get("/list-documents/", dependencies=[Depends(api_key_auth)])
def list_documents(course: str = None):
    try:
        query_response = pinecone_index.query(
            vector=[0.0] * 768,
            top_k=10000,
            include_metadata=True
        )
        documents = {}
        for match in query_response['matches']:
            metadata = match['metadata']
            if 'source' in metadata and 'course' in metadata:
                doc_course = metadata['course']
                doc_source = metadata['source']
                
                if course and doc_course != course:
                    continue
                
                if doc_course not in documents:
                    documents[doc_course] = set()
                documents[doc_course].add(doc_source)
        
        result = {course: sorted(list(docs)) for course, docs in documents.items()}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao listar documentos: {str(e)}")


@app.post("/upload-and-process/", dependencies=[Depends(api_key_auth)])
async def upload_and_process(
    file: UploadFile = File(...), 
    doc_name: str = Form(...),
    course: str = Form(...)
):
    allowed_content_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain", "text/markdown"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(status_code=400, detail=f"Tipo de arquivo '{file.content_type}' não suportado.")

    if not course or course.strip() == "":
        raise HTTPException(status_code=400, detail="O nome do curso é obrigatório.")

    existing_docs = list_documents(course=course)
    if course in existing_docs and doc_name in existing_docs[course]:
        raise HTTPException(status_code=400, detail=f"Um documento com o nome '{doc_name}' já existe no curso '{course}'.")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        raw_text = process_pdf_text(tmp_path)
        chunks = create_text_chunks(raw_text)
        
        metadatas = [{"source": doc_name, "course": course} for _ in chunks]
        LangchainPinecone.from_texts(
            texts=chunks, 
            embedding=embeddings_model, 
            metadatas=metadatas, 
            index_name=PINECONE_INDEX_NAME
        )
        
        return {
            "status": "sucesso", 
            "filename": doc_name, 
            "course": course,
            "message": f"Documento '{doc_name}' processado e adicionado ao curso '{course}'."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro: {str(e)}")
    finally:
        os.remove(tmp_path)


class DocumentRequest(BaseModel):
    filename: str
    course: str 


@app.post("/delete-document/", dependencies=[Depends(api_key_auth)])
def delete_document(request: DocumentRequest):
    try:
        pinecone_index.delete(filter={
            "source": {"$eq": request.filename},
            "course": {"$eq": request.course}
        })
        return {
            "status": "sucesso", 
            "filename": request.filename,
            "course": request.course,
            "message": f"Documento '{request.filename}' removido do curso '{request.course}'."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao remover documento: {str(e)}")


class QuestionRequest(BaseModel):
    question: str
    course: str 
    chat_history: list = []


@app.post("/ask/", dependencies=[Depends(api_key_auth)])
def ask_question(request: QuestionRequest):
    try:
        vectorstore = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME, 
            embedding=embeddings_model
        )
        
        retriever = vectorstore.as_retriever(
            search_kwargs={"filter": {"course": {"$eq": request.course}}}
        )
        
        formatted_history = []
        for i in range(0, len(request.chat_history), 2):
            if i + 1 < len(request.chat_history):
                user_msg = request.chat_history[i]
                assistant_msg = request.chat_history[i+1]
                if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                    formatted_history.append((user_msg["content"], assistant_msg["content"]))

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        response = conversation_chain.invoke({
            'question': request.question, 
            'chat_history': formatted_history
        })

        
        source_documents_formatted = []
        if 'source_documents' in response:
            for doc in response['source_documents']:
                print(response['source_documents'])
                source_documents_formatted.append({
                    "page_content": doc.page_content,
                    "source": doc.metadata.get('source', 'N/A'),
                    "course": doc.metadata.get('course', 'N/A')
                })
        
        return {
            "answer": response['answer'], 
            "source_documents": source_documents_formatted
        }

    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar pergunta: {str(e)}")