import os
import torch
from typing import List
from flask import Blueprint, request, jsonify
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from app.routes.extractor import extract_data
from app.routes.topic_extractor import extract_from_headers_list, extract_role_of_committee_subsections

# Load environment variables
load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./_chromadb")

embedding_model = OllamaEmbeddings(model = "mxbai-embed-large")

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create Blueprint
vectordb_bp = Blueprint('vectordb', __name__)


@vectordb_bp.route('/', methods=['GET'])
def home():
    return jsonify({"message": "getting started"}), 200

@vectordb_bp.route('/', methods=['POST'])
def rag():
    pdf_folder = "data"

    files =[]
    
    for file in os.listdir(pdf_folder):

        if file.lower().endswith(".pdf"):

            files.append(file)
            print(file + '\n')

    final_chunks=[]

    for file in files:
        extracted = []

    # Extract special subsections
        if file == 'CBL_Nomination_and_Remuneration_Policy.pdf':

            result = extract_data(pdf_folder, file)

            extracted = extract_from_headers_list(result, file)

            special_extract = extract_role_of_committee_subsections(extracted[2]['content'], file)

            extracted.pop(2)

            extracted.extend(special_extract)

        elif file == 'Inactive_TradingAccount_Policy_version3.pdf':

            result = extract_data(pdf_folder, file)

            extracted = extract_from_headers_list(result, file)

            special_extract = extract_role_of_committee_subsections(extracted[1]['content'], file)

            extracted.pop(1)

            extracted.extend(special_extract)

        else:

            print(f"Extracting from {file}")

            result = extract_data(pdf_folder, file)

            extracted = extract_from_headers_list(result, file)

            print(f"{file} extracted.\n\n")
        
        # Chunk and store
        prepared_chunks = prepare_chunks(extracted)

        final_chunks.extend(prepared_chunks)

    for data in final_chunks:
        print(f"metadata : {data['metadata']}\n")
        print(f"content : {data['content']}\n\n")
    store_to_chroma(final_chunks)

    return jsonify({
        "success": "Vector DB created and data stored.",
        "chunks_saved": len(final_chunks)
    }), 201
   

    store_to_chroma(final_chunks)

    return jsonify({
        "success": "Vector DB created and data stored.",
        "chunks_saved": len(final_chunks)
    }), 201
 
# --- Helpers ---

def prepare_chunks(data_list, chunk_size=800, overlap=128):
    chunks = []
    for item in data_list:
        content = item["content"]
        metadata = item["metadata"]
        topic = metadata.get("topic", "")
        subtopic = metadata.get("subtopic", "")
        page = metadata.get("page_no", 1)

        prefix = f"Topic: {topic}\nSubtopic: {subtopic}\n\n"

        if len(content) <= chunk_size:
            chunks.append({
                "content": prefix + content,
                "metadata": {**metadata, "page_no": page}
            })
        else:
            start = 0
            while start < len(content):
                end = start + chunk_size
                chunk = content[start:end]
                chunks.append({
                    "content": prefix + chunk,
                    "metadata": {**metadata, "page_no": page}
                })
                start += chunk_size - overlap

    return chunks

def store_to_chroma(chunks: List[dict]):
    documents = [
        Document(page_content=c["content"], metadata=c["metadata"])
        for c in chunks
    ]

    Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        
        collection_name="pdf_collection"
    )

    print("Documents successfully embedded and stored in Chroma.")
