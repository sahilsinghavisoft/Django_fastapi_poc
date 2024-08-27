import os
import io
import re
import time
import pickle
import django
import logging
import pdfplumber
import numpy as np

from openai import OpenAI
from mongoengine import connect
from asgiref.sync import sync_to_async
from model import TeacherUpload, Question

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import FastAPI, UploadFile, Response, Request, File, Query, HTTPException

from jinja2 import Environment, FileSystemLoader

app = FastAPI()

# Configuring Django settings and initializing Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

from django.contrib.sessions.models import Session
from django.contrib.auth import get_user_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@sync_to_async
def get_user_from_session_sync(session_key):
    try:
        logger.info(f"Attempting to retrieve session with key: {session_key}")
        session = Session.objects.get(session_key=session_key)
        session_data = session.get_decoded()
        user_id = session_data.get('_auth_user_id')
        user_model = get_user_model()
        user = user_model.objects.get(id=user_id)
        logger.info(f"User found: {user.username}")
        return user
    except Session.DoesNotExist:
        logger.error("Session does not exist")
        return None
    except user_model.DoesNotExist:
        logger.error("User does not exist")
        return None

# MongoDB setup
connect(db="question_pdf_container", host="localhost", port=27017)

app.mount("/static", StaticFiles(directory="static"), name="static")
template_env = Environment(loader=FileSystemLoader("static"))

@app.get("/home")
async def home(request: Request, response: Response):
    return RedirectResponse(url="http://127.0.0.1:8000/", status_code=303)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    session_key = request.cookies.get('sessionid')
    if not session_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user = await get_user_from_session_sync(session_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if user.is_student:
        role = 'student'
    elif user.is_teacher:
        role = 'teacher'
    elif user.is_superuser:
        role = 'superuser'
    else:
        role = 'guest'

    template = template_env.get_template("dashboard.html")
    html_content = template.render(role=role)

    return HTMLResponse(content=html_content, media_type="text/html")

@app.api_route("/upload-pdf", methods=["GET", "POST"], response_class=HTMLResponse)
async def upload_pdf(request: Request, pdf: UploadFile = File(None)):
    session_key = request.cookies.get('sessionid')
    user = await get_user_from_session_sync(session_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if user.is_superuser or user.is_teacher:
        if request.method == "GET":
            with open(os.path.join("static", "upload_pdf.html")) as f:
                return HTMLResponse(content=f.read(), media_type="text/html")
        
        elif request.method == "POST":
            if pdf is None:
                raise HTTPException(status_code=400, detail="No file provided")
            
            if pdf.content_type != "application/pdf":
                raise HTTPException(status_code=400, detail="File is not a PDF")

            teacher = user.username
            pdf_content = await pdf.read()

            teacher_upload = TeacherUpload(
                user_id=user.id,
                teacher=teacher,
                pdf_file=pdf_content,
            )
            teacher_upload.save()

            return JSONResponse(content={"message": "PDF uploaded successfully", "teacher": teacher})
    else:
        raise HTTPException(status_code=403, detail="Access Denied")

@app.route("/process-all-pdfs/", methods=["GET", "POST"])
async def process_all_pdfs(request: Request):
    session_key = request.cookies.get('sessionid')
    user = await get_user_from_session_sync(session_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if user.is_superuser:
        if request.method == "GET":
            with open(os.path.join("static", "process_all_pdfs.html")) as f:
                return HTMLResponse(content=f.read(), media_type="text/html")
        
        elif request.method == "POST":
            start_time = time.time()

            uploads = TeacherUpload.objects()
            if not uploads:
                raise HTTPException(status_code=404, detail="No PDFs found")

            model = SentenceTransformer('all-MiniLM-L6-v2')
            all_chunks = []

            for upload in uploads:
                pdf_file = upload.pdf_file.read()
                with pdfplumber.open(io.BytesIO(pdf_file)) as pdf_doc:
                    text = "".join(page.extract_text() for page in pdf_doc.pages if page.extract_text())
                
                all_chunks.extend(adaptive_chunking(text, model))

            if not all_chunks:
                raise HTTPException(status_code=404, detail="No text chunks created from PDFs")

            original_store_name = "pdf_collection"
            store_name = re.sub(r'[^a-zA-Z0-9_]', '_', original_store_name)
            collection = create_milvus_collection(store_name)
            
            embeddings = model.encode(all_chunks, show_progress_bar=True)
            insert_embeddings(collection, embeddings, all_chunks)
            create_index(collection)

            metadata = {
                'collection_name': store_name,
                'embedding_dimension': 384
            }
            pkl_filename = f"{store_name}.pkl"
            with open(pkl_filename, "wb") as f:
                pickle.dump(metadata, f)

            end_time = time.time()
            logger.info(f"Processed PDFs in {end_time - start_time:.2f} seconds")
            return JSONResponse(content={"store_name": store_name, "processing_time": end_time - start_time})
    else:
        raise HTTPException(status_code=403, detail="Access Denied")

@app.get("/generate-response/", response_class=HTMLResponse)
async def generate_response(request: Request, query: str = Query(None)):
    session_key = request.cookies.get('sessionid')
    user = await get_user_from_session_sync(session_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if user.is_superuser or user.is_teacher or user.is_student:
        if not query:
            with open(os.path.join("static", "generate_response.html")) as f:
                return HTMLResponse(content=f.read(), media_type="text/html")

        model = SentenceTransformer('all-MiniLM-L6-v2')
        k = 5
        results = []

        for store_name in os.listdir('.'):
            if store_name.endswith('.pkl'):
                try:
                    with open(store_name, "rb") as f:
                        metadata = pickle.load(f)

                    collection_name = metadata.get('collection_name')
                    if not collection_name:
                        logger.warning(f"Invalid metadata in {store_name}")
                        continue

                    connect_to_milvus()
                    collection = Collection(name=collection_name)

                    create_index(collection)

                    query_embedding = model.encode([query])
                    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
                    search_results = collection.search(query_embedding.tolist(), "embedding", search_params, limit=k, output_fields=["metadata"])

                    docs = [
                        SimpleDocument(hit.entity.metadata["text"], {"score": hit.distance})
                        for hit in search_results[0]
                    ]
                    results.extend(docs)
                except Exception as e:
                    logger.error(f"Error processing collection from {store_name}: {e}")
                    continue

        if not results:
            return JSONResponse(content={"answer": "No relevant information found in any collection."})

        results.sort(key=lambda x: x.metadata['score'])
        top_results = results[:k]

        context = "\n\n".join([f"Relevance: {doc.metadata['score']:.4f}\nContent: {doc.page_content}" for doc in top_results])

        prompt = (
            "You are a helpful AI assistant. Your task is to answer the user's query based on the provided context. \n"
            "Follow these guidelines:\n"
            "1. Use only the information from the given context to answer the query.\n"
            "2. If the context doesn't contain relevant information, say \"I don't have enough information to answer that question.\"\n"
            "3. Cite the relevance scores when referring to specific pieces of information.\n"
            "4. Provide a concise yet informative answer.\n"
            "5. If appropriate, suggest follow-up questions or areas for further exploration.\n\n"
            f"### Context:\n{context}\n\n"
            f"### User Query:\n{query}\n\n"
            "### Response:\n"
        )

        try:
            response = client.chat.completions.create(
                model="TechxGenus/Meta-Llama-3-8B-Instruct-AWQ",
                messages=[
                    {"role": "system", "content": "You are an AI assistant specializing in providing accurate and helpful information based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

        student = user.username
        question_entry = Question(
            user_id=user.id,
            student=student,
            question_text=query,
            answer_text=answer
        )
        question_entry.save()

        logger.info(f"Generated response for query: {query} and saved to database")
        return JSONResponse(content={"answer": answer})
    else:
        raise HTTPException(status_code=403, detail="Access Denied")

# Utility class
class SimpleDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

# Milvus connection and collection setup
def connect_to_milvus(host="localhost", port=19530):
    connections.connect("default", host=host, port=19530)
    logger.info("Connected to Milvus")

def create_milvus_collection(collection_name):
    connect_to_milvus()

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
    schema = CollectionSchema(fields=fields, description="Collection for document embeddings")
    collection = Collection(name=collection_name, schema=schema)
    logger.info(f"Created Milvus collection: {collection_name}")
    return collection

def insert_embeddings(collection, embeddings, metadata):
    entities = [
        {
            "embedding": [float(value) for value in embedding.tolist()],
            "metadata": {"text": chunk}
        }
        for embedding, chunk in zip(embeddings, metadata)
    ]
    collection.insert(entities)
    logger.info(f"Inserted {len(embeddings)} embeddings into Milvus")

def create_index(collection):
    if not collection.has_index():
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        logger.info(f"Created index for collection {collection.name}")
    collection.load()
    logger.info(f"Loaded collection {collection.name}")

def adaptive_chunking(text, model, chunk_size_range=(256, 512), overlap_range=(50, 100), similarity_threshold=0.7):
    sentences = text.split('.')  # Simple sentence splitting
    embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk = []
    chunk_size = chunk_size_range[0]
    overlap = overlap_range[0]
    
    for i, embedding in enumerate(embeddings):
        if current_chunk:
            last_embedding = model.encode([' '.join(current_chunk)])  # Encode current chunk
            similarity = cosine_similarity([embedding], last_embedding)[0][0]
            if similarity < similarity_threshold:
                chunks.append(' '.join(current_chunk))
                current_chunk = current_chunk[-overlap:]  # Keep overlap
                chunk_size = min(chunk_size + 50, chunk_size_range[1])
                overlap = min(overlap + 10, overlap_range[1])
        
        current_chunk.append(sentences[i].strip())
        if len(current_chunk) >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-overlap:]
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# OpenAI client setup
client = OpenAI(
    base_url="http://54.210.171.24:8000/v1",
    api_key="YOUR_OPENAI_API_KEY"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)