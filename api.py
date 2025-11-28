import os
import tempfile
import re
import fitz 
from datetime import datetime, timedelta
from typing import Annotated, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Form, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.chains import ConversationalRetrievalChain
from sqlalchemy.orm import Session
from sqlalchemy import func
from jose import JWTError, jwt
from passlib.context import CryptContext
import pinecone

from database import get_db, ChatLog, User, SessionLocal, Base, engine

Base.metadata.create_all(bind=engine)

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET", "312323xkokpsl")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.5)

app = FastAPI(title="API do Chat RAG")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# func de autenticação
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.on_event("startup")
def create_initial_admin():
    db = SessionLocal()
    try:
        admin_user = os.getenv("ADMIN_USERNAME", "admin")
        admin_pass = os.getenv("ADMIN_PASSWORD", "admin123")
        
        user = db.query(User).filter(User.username == admin_user).first()
        if not user:
            print(f"⚙️ Criando Admin: '{admin_user}'")
            hashed_pwd = get_password_hash(admin_pass)
            db_admin = User(username=admin_user, hashed_password=hashed_pwd, is_admin=True)
            db.add(db_admin)
            db.commit()
            print("✅ Admin criado com sucesso!")
        else:
            print("✅ Admin já existe.")
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: 
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None: 
        raise credentials_exception
    return user

async def get_current_admin(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Acesso negado: Requer privilégios de Admin")
    return current_user

# func internas 
def _get_documents_by_course(course: str = None) -> dict:
    try:
        query_response = pinecone_index.query(
            vector=[0.0] * 768,
            top_k=10000,
            include_metadata=True
        )
        
        documents = {}
        for match in query_response['matches']:
            metadata = match.get('metadata', {})
            
            if 'source' not in metadata or 'course' not in metadata:
                continue
                
            doc_course = metadata['course']
            doc_source = metadata['source']
            
            if course and doc_course != course:
                continue
            
            if doc_course not in documents:
                documents[doc_course] = set()
            
            documents[doc_course].add(doc_source)
        
        result = {
            course: sorted(list(docs)) 
            for course, docs in documents.items()
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao listar documentos: {str(e)}"
        )

def _get_courses_list() -> List[str]:
    try:
        query_response = pinecone_index.query(
            vector=[0.0] * 768,
            top_k=10000,
            include_metadata=True
        )
        courses = set()
        for match in query_response['matches']:
            if 'course' in match.get('metadata', {}):
                courses.add(match['metadata']['course'])
        return sorted(list(courses))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao listar cursos: {str(e)}")

def process_pdf_text(pdf_path):
    full_text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                full_text += page.get_text("text") + "\n\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler PDF: {e}")
    
    full_text = re.sub(r' +', ' ', full_text)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    return full_text

def create_text_chunks(full_text):
    text_splitter = CharacterTextSplitter(
        separator='\n', 
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    return text_splitter.split_text(full_text)

# endpoints de auth
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Usuário ou senha incorretos")
    
    access_token = create_access_token(data={"sub": user.username})
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "is_admin": user.is_admin
    }

@app.post("/register")
def register_student(
    username: str = Form(...), 
    password: str = Form(...), 
    db: Session = Depends(get_db)
):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Usuário já existe")
    
    hashed_pwd = get_password_hash(password)
    new_user = User(username=username, hashed_password=hashed_pwd, is_admin=False)
    db.add(new_user)
    db.commit()
    return {"message": "Conta criada com sucesso"}

# endpoints de stats
class StatsOverview(BaseModel):
    total_questions: int
    total_courses: int
    total_vectors: int

@app.get("/stats/overview", response_model=StatsOverview)
def get_stats_overview(
    db: Session = Depends(get_db), 
    user: User = Depends(get_current_admin)
):
    try:
        total_questions = db.query(func.count(ChatLog.id)).scalar()
        
        try:
            resp = pinecone_index.query(
                vector=[0.0]*768, 
                top_k=10000, 
                include_metadata=True
            )
            courses = set(
                m['metadata']['course'] 
                for m in resp['matches'] 
                if 'course' in m.get('metadata', {})
            )
            total_courses = len(courses)
        except:
            total_courses = 0
        
        index_stats = pinecone_index.describe_index_stats()
        total_vectors = index_stats.get('total_vector_count', 0)
        
        return {
            "total_questions": total_questions,
            "total_courses": total_courses,
            "total_vectors": total_vectors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro stats: {str(e)}")

@app.get("/stats/questions-by-course")
def get_questions_by_course(
    db: Session = Depends(get_db), 
    user: User = Depends(get_current_admin)
):
    try:
        result = db.query(
            ChatLog.course, 
            func.count(ChatLog.id).label('question_count')
        ).group_by(ChatLog.course).order_by(func.count(ChatLog.id).desc()).all()
        
        return {row[0]: row[1] for row in result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro stats por curso: {str(e)}")

@app.get("/stats/recent-questions")
def get_recent_questions(
    limit: int = 20, 
    db: Session = Depends(get_db), 
    user: User = Depends(get_current_admin)
):
    try:
        logs = db.query(ChatLog).order_by(ChatLog.timestamp.desc()).limit(limit).all()
        return logs 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro recentes: {str(e)}")

# endpoints de consulta 
@app.get("/list-courses/", response_model=List[str])
def list_courses(user: User = Depends(get_current_user)):
    return _get_courses_list()

@app.get("/list-documents/")
def list_documents(
    course: str = None, 
    user: User = Depends(get_current_user)
):
    return _get_documents_by_course(course)

# endpoints de gestao de documentos 
@app.post("/upload-and-process/")
async def upload_and_process(
    file: UploadFile = File(...), 
    doc_name: str = Form(...),
    course: str = Form(...),
    user: User = Depends(get_current_admin)
):
    
    allowed_content_types = ["application/pdf"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Tipo '{file.content_type}' não suportado. Apenas PDF."
        )

    if not course or course.strip() == "":
        raise HTTPException(status_code=400, detail="Nome do curso é obrigatório.")

    existing_docs = _get_documents_by_course(course=course)
    if course in existing_docs and doc_name in existing_docs[course]:
        raise HTTPException(
            status_code=400, 
            detail=f"Documento '{doc_name}' já existe no curso '{course}'."
        )

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
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")
    finally:
        os.remove(tmp_path)

class DocumentRequest(BaseModel):
    filename: str
    course: str 

@app.post("/delete-document/")
def delete_document(
    request: DocumentRequest, 
    user: User = Depends(get_current_admin)
):
    
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
        raise HTTPException(status_code=500, detail=f"Erro remover: {str(e)}")

# endpoint de conversação RAG
class QuestionRequest(BaseModel):
    question: str
    course: str 
    chat_history: list = []

@app.post("/ask/")
def ask_question(
    request: QuestionRequest, 
    db: Session = Depends(get_db), 
    user: User = Depends(get_current_user)
):
    try:
        vectorstore = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME, 
            embedding=embeddings_model
        )
        
        retriever = vectorstore.as_retriever(
            search_kwargs={"filter": {"course": {"$eq": request.course}}, "k": 4}
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

        try:
            new_log = ChatLog(
                course=request.course,
                question=request.question,
                answer=response['answer'],
                timestamp=datetime.now()
            )
            db.add(new_log)
            db.commit()
        except Exception as e:
            print(f"⚠️ Erro ao salvar log no DB: {e}") 
            db.rollback()
        
        print(response)

        source_documents_formatted = []
        if 'source_documents' in response:
            print(f"📄 Documentos recuperados: {len(response['source_documents'])}")
            for doc in response['source_documents']:
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

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Chat RAG API",
        "version": "2.0",
        "auth": "JWT"
    }