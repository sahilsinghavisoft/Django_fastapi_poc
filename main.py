import os
import pickle
import time
import io
import re
from sentence_transformers import SentenceTransformer
import pdfplumber
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from openai import OpenAI
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SimpleDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

def connect_to_milvus(host="localhost", port=19530):
    connections.connect("default", host=host, port=port)
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

client = OpenAI(
    base_url="http://3.82.136.181:8000/v1",
    api_key="YOUR_OPENAI_API_KEY"
)

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

@app.post("/upload-pdf/")
async def upload_pdf(pdf: UploadFile = File(...)):
    start_time = time.time()

    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File is not a PDF")

    pdf_content = await pdf.read()
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf_doc:
        text = "".join(page.extract_text() for page in pdf_doc.pages if page.extract_text())

    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunks = adaptive_chunking(text, model)

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
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(metadata, f)

    end_time = time.time()
    logger.info(f"Processed PDF: {pdf.filename} in {end_time - start_time:.2f} seconds")
    return JSONResponse(content={"store_name": store_name, "processing_time": end_time - start_time})

@app.get("/generate_response/")
async def generate_response(query: str = Query(...)):
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

    # Sort results by relevance score
    results.sort(key=lambda x: x.metadata['score'])
    top_results = results[:k]

    # Create a more dynamic context
    context = "\n\n".join([f"Relevance: {doc.metadata['score']:.4f}\nContent: {doc.page_content}" for doc in top_results])

    # Enhanced prompt generation
    prompt = (
        "You are a helpful AI assistant. Your task is to answer the user's query based on the provided context. \n"
        "Follow these guidelines:\n"
        "1. Use only the information from the given context to answer the query.\n"
        "2. If the context doesn't contain relevant information, say \"I don't have enough information to answer that question.\"\n"
        "3. Cite the relevance scores when referring to specific pieces of information.\n"
        "4. Provide a concise yet informative answer.\n\n"
        f"### Context:\n{context}\n\n"
        f"### User Query:\n{query}\n\n"
        "### Response:\n"
    )

    try:
        response = client.chat.completions.create(
            model="TechxGenus/Meta-Llama-3-8B-Instruct-AWQ",
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,  # Increased token limit for more detailed responses
            temperature=0.7,  # Adjust temperature for more creative responses
            top_p=0.9,  # Use nucleus sampling for more diverse responses
            frequency_penalty=0.5,  # Reduce repetition
            presence_penalty=0.5  # Encourage new topics
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    logger.info(f"Generated response for query: {query}")
    return JSONResponse(content={"answer": answer})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)