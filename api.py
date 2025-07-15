import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Importações de LangChain e outras
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import Pinecone  # <<< MUDANÇA IMPORTANTE NA IMPORTAÇÃO
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# --- INICIALIZAÇÃO E CONFIGURAÇÃO ---

# Carrega as variáveis de ambiente do arquivo .env (para rodar localmente)
# No Render, ele usará as variáveis que você configurou no painel.
load_dotenv()

# Configura as chaves de API como variáveis de ambiente para que as bibliotecas as encontrem
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# Pega o nome do índice do Pinecone do ambiente
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Carrega os modelos de IA (usando o cache do Streamlit, mas aqui eles serão carregados na inicialização da API)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)

# Inicializa a aplicação FastAPI
app = FastAPI(title="API do Chat RAG")


# --- FUNÇÕES DE APOIO (PROCESSAMENTO DE PDF) ---

def processar_texto_de_pdf(pdf_file):
    """Lê o conteúdo de um arquivo PDF e retorna como texto."""
    texto = ""
    try:
        leitor_pdf = PdfReader(pdf_file)
        for pagina in leitor_pdf.pages:
            texto_pagina = pagina.extract_text()
            if texto_pagina:
                texto += texto_pagina
    except Exception as e:
        # Em caso de erro, lança uma exceção HTTP que o FastAPI entende.
        raise HTTPException(status_code=500, detail=f"Erro ao ler o arquivo PDF: {e}")
    return texto


def criar_chunks(texto):
    """Quebra um texto longo em pedaços menores (chunks)."""
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(texto)


# --- ENDPOINTS DA API (AS "PORTAS" DE ENTRADA) ---

@app.post("/upload-and-process/")
async def upload_and_process(file: UploadFile = File(...)):
    """
    Endpoint para receber um arquivo PDF, processá-lo e adicionar o conhecimento
    ao banco de dados de vetores no Pinecone.
    """
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Tipo de arquivo inválido. Por favor, envie um PDF.")

    try:
        # Salva o arquivo enviado em um local temporário para poder ser lido.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Usa as funções de apoio para processar o PDF.
        raw_text = processar_texto_de_pdf(tmp_path)
        chunks = criar_chunks(raw_text)

        # Adiciona os textos processados ao índice do Pinecone.
        # O LangChain lida com a criação dos embeddings e o envio.
        Pinecone.from_texts(texts=chunks, embedding=embeddings, index_name=PINECONE_INDEX_NAME)

        # Apaga o arquivo temporário após o uso.
        os.remove(tmp_path)

        return {"status": "sucesso", "filename": file.filename,
                "message": "Documento processado e adicionado à base de conhecimento."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro inesperado: {str(e)}")


# Define o formato esperado para a requisição de pergunta
class QuestionRequest(BaseModel):
    question: str
    chat_history: list = []


@app.post("/ask/")
def ask_question(request: QuestionRequest):
    """
    Endpoint para receber uma pergunta, buscar o contexto no Pinecone,
    e gerar uma resposta com o Gemini.
    """
    try:
        # Conecta ao índice Pinecone que já existe.
        vectorstore = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

        # Cria a cadeia de conversação para esta requisição específica.
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True
        )

        # Executa a cadeia com a pergunta e o histórico para obter a resposta.
        response = conversation_chain({'question': request.question, 'chat_history': request.chat_history})
        answer = response['answer']

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao processar a pergunta: {str(e)}")