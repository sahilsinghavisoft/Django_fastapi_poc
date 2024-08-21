import os
import pickle
import time
import io
import re
from sentence_transformers import SentenceTransformer
import pdfplumber
from fastapi import FastAPI, UploadFile, Request, File, Query, HTTPException
from fastapi.responses import JSONResponse
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfplumber import open as pdfplumber_open
from openai import OpenAI
import django
import logging
from model import TeacherUpload, Question
from mongoengine import connect
from asgiref.sync import sync_to_async
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

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

@app.get("/upload-pdf-ui", response_class=HTMLResponse)
async def upload_pdf_ui():
    with open(os.path.join("static", "upload_pdf.html")) as f:
        return HTMLResponse(content=f.read(), media_type="text/html")

@app.get("/generate-response-ui", response_class=HTMLResponse)
async def generate_response_ui():
    with open(os.path.join("static", "generate_response.html")) as f:
        return HTMLResponse(content=f.read(), media_type="text/html")


# Utility class
class SimpleDocument:
    def _init_(self, page_content, metadata=None):
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
            "embedding": [float(value) for value in embedding.tolist()],  # Convert to float
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

# OpenAI client setup
client = OpenAI(
    base_url="http://3.82.136.181:8000/v1",
    api_key="YOUR_OPENAI_API_KEY"
)

@app.post("/upload-pdf/")
async def upload_pdf(request: Request, pdf: UploadFile = File(...)):
    start_time = time.time()

    # Retrieve session key from request (e.g., from cookies)
    session_key = request.cookies.get('sessionid')
    user = await get_user_from_session_sync(session_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File is not a PDF")

    pdf_content = await pdf.read()
    with pdfplumber_open(io.BytesIO(pdf_content)) as pdf_doc:
        text = "".join(page.extract_text() for page in pdf_doc.pages if page.extract_text())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    original_store_name = pdf.filename[:-4]
    store_name = re.sub(r'[^a-zA-Z0-9_]', '_', original_store_name)

    collection = create_milvus_collection(store_name)

    embeddings = model.encode(chunks, show_progress_bar=True)
    insert_embeddings(collection, embeddings, chunks)
    create_index(collection)

    metadata = {
        'collection_name': store_name,
        'embedding_dimension': 384
    }
    pkl_filename = f"{store_name}.pkl"
    with open(pkl_filename, "wb") as f:
        pickle.dump(metadata, f)

    # Save the PDF and pickle file to the TeacherUpload model
    teacher = user.username  # Get the teacher's username from the user object
    teacher_upload = TeacherUpload(
        user_id=user.id,  # Store the session ID
        teacher=teacher,  # Store the teacher's name
        pdf_file=pdf_content,  # Save the raw PDF file
        pkl_file=open(pkl_filename, "rb").read()  # Save the pickle file as binary
    )
    teacher_upload.save()

    end_time = time.time()
    logger.info(f"Processed PDF: {pdf.filename} in {end_time - start_time:.2f} seconds")
    return JSONResponse(content={"store_name": store_name, "processing_time": end_time - start_time})

@app.get("/generate_response/")
async def generate_response(request: Request, query: str = Query(...)):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    k = 5
    results = []

    # Retrieve session key from request (e.g., from cookies)
    session_key = request.cookies.get('sessionid')
    user = await get_user_from_session_sync(session_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # for store_name in os.listdir('.'):
    #     if store_name.endswith('.pkl'):
    #         try:
    #             with open(store_name, "rb") as f:
    #                 metadata = pickle.load(f)

    #             collection_name = metadata.get('collection_name')
    #             if not collection_name:
    #                 logger.warning(f"Invalid metadata in {store_name}")
    #                 continue

    #             connect_to_milvus()
    #             collection = Collection(name=collection_name)

    #             create_index(collection)

    #             query_embedding = model.encode([query])
    #             search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    #             search_results = collection.search(query_embedding.tolist(), "embedding", search_params, limit=k, output_fields=["metadata"])

    #             docs = [
    #                 SimpleDocument(hit.entity.metadata["text"], {"score": hit.distance})
    #                 for hit in search_results[0]
    #             ]
    #             results.extend(docs)
    #         except Exception as e:
    #             logger.error(f"Error processing collection from {store_name}: {e}")
    #             continue

    # if not results:
    #     return JSONResponse(content={"answer": "No relevant information found in any collection."})

    # # Sort results by relevance score
    # results.sort(key=lambda x: x.metadata['score'])
    # top_results = results[:k]

    # # Create a more dynamic context
    # context = "\n\n".join([f"Relevance: {doc.metadata['score']:.4f}\nContent: {doc.page_content}" for doc in top_results])

    # # Enhanced prompt generation
    # prompt = (
    #     "You are a helpful AI assistant. Your task is to answer the user's query based on the provided context. \n"
    #     "Follow these guidelines:\n"
    #     "1. Use only the information from the given context to answer the query.\n"
    #     "2. If the context doesn't contain relevant information, say \"I don't have enough information to answer that question.\"\n"
    #     "3. Cite the relevance scores when referring to specific pieces of information.\n"
    #     "4. Provide a concise yet informative answer.\n\n"
    #     f"### Context:\n{context}\n\n"
    #     f"### User Query:\n{query}\n\n"
    #     "### Response:\n"
    # )

    # try:
    #     response = client.chat.completions.create(
    #         model="TechxGenus/Meta-Llama-3-8B-Instruct-AWQ",
    #         messages=[
    #             {"role": "system", "content": "You are an AI assistant."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         max_tokens=600,
    #         temperature=0.7,
    #         top_p=0.9,
    #         frequency_penalty=0.5,
    #         presence_penalty=0.5
    #     )
    #     # answer = response.choices[0].message.content.strip()
    #     answer="Hey its me!"
    # except Exception as e:
    #     logger.error(f"Error generating response: {e}")
    #     raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    answer=f"Hey its me {user.username}!"

    # Save the query and answer to the Question model
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

if __name__ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)