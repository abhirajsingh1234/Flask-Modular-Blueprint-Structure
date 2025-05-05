# import os
# import base64
# from typing import List
# from flask import Blueprint, request, jsonify
# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# import torch

# load_dotenv()
# CHROMA_DIR = os.getenv("CHROMA_DIR", "./_chromadb")
# #from app.extensions import db
# device = "cuda" if torch.cuda.is_available() else "cpu"
# vectordb_bp = Blueprint('vectordb', __name__)
# embeddings = HuggingFaceEmbeddings(
#     model_name="intfloat/e5-large-v2",
#     model_kwargs={"device": "cuda"},
#     encode_kwargs={"normalize_embeddings":True}
# )

# @vectordb_bp.route('/', methods=['GET'])
# def home():
#     return jsonify({"message":"getting started"}), 200

# @vectordb_bp.route('/', methods=['POST'])
# def rag():
#     pdf_folder = "data"
#     pdf_files = [
#         os.path.join(pdf_folder, fn)
#         for fn in os.listdir(pdf_folder)
#         if fn.lower().endswith(".pdf")
#     ]
#     raw_docs: List[Document] = []
#     basename_list: List = []  
#     for path in pdf_files:
#         basename = os.path.basename(path)
#         loader = UnstructuredPDFLoader(path)
#         pages  = loader.load()
#         for d in pages:
#             d.metadata["source"] = basename
#             basename_list.append(basename)
#             raw_docs.append(d)
#     # serializable_docs = [
#     #     {
#     #         "text": d.page_content,       # or slice d.page_content[:200] if it’s huge
#     #         "metadata": d.metadata
#     #     }
#     #     for d in raw_docs
#     # ]

#     # custom split function by abhiraj
    
#     splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=128)

#     chunked_docs: List[Document] = []
#     for doc in raw_docs:
#         texts = splitter.split_text(doc.page_content)
#         for idx, txt in enumerate(texts):
#             chunked_docs.append(
#                 Document(
#                     page_content=txt,
#                     metadata={
#                         "source": doc.metadata["source"],
#                         "chunk":  idx
#                     }
#                 )
#             )
#     print(chunked_docs[0], chunked_docs[1], chunked_docs[3])

#     vectordb = Chroma.from_documents(
#         documents=chunked_docs,
#         embedding=embeddings,
#         persist_directory=CHROMA_DIR
#     )
    

#     return jsonify({"success":"vector db created",
#                     "retrieved document":basename_list}), 201

import os
import torch
from typing import List
from flask import Blueprint, request, jsonify
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from functools import lru_cache
import concurrent.futures

# Load environment variables
load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./_chromadb")
DATA_DIR = os.getenv("DATA_DIR", "data")

# Create Blueprint
vectordb_bp = Blueprint('vectordb', __name__)

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1)
def get_embeddings():
    """Returns cached embedding model to avoid reloading"""
    return HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

def process_pdf(pdf_path):
    """Process a single PDF file into chunked documents"""
    basename = os.path.basename(pdf_path)
    raw_docs = []
    
    try:
        # Load PDF
        loader = UnstructuredPDFLoader(pdf_path)
        pages = loader.load()
        
        # Add source metadata
        for d in pages:
            d.metadata["source"] = basename
            raw_docs.append(d)
            
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=128)
        
        chunked_docs = []
        for doc in raw_docs:
            texts = splitter.split_text(doc.page_content)
            for idx, txt in enumerate(texts):
                chunked_docs.append(
                    Document(
                        page_content=txt,
                        metadata={
                            "source": doc.metadata["source"],
                            "chunk": idx
                        }
                    )
                )
                
        return chunked_docs
    except Exception as e:
        print(f"Error processing {basename}: {str(e)}")
        return []


@vectordb_bp.route('/', methods=['GET'])
def rag():
    # Get all PDF files
    pdf_files = [
        os.path.join(DATA_DIR, fn)
        for fn in os.listdir(DATA_DIR)
        if fn.lower().endswith(".pdf")
    ]
    
    if not pdf_files:
        return jsonify({"error": "No PDF files found in data directory"}), 400
    
    # Process PDFs in parallel using a thread pool
    all_docs = []
    basename_list = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all PDF processing tasks
        future_to_pdf = {executor.submit(process_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            basename = os.path.basename(pdf_path)
            
            try:
                docs = future.result()
                if docs:
                    all_docs.extend(docs)
                    basename_list.append(basename)
            except Exception as e:
                print(f"Error processing {basename}: {str(e)}")
    
    # Check if we have documents to process
    if not all_docs:
        return jsonify({"error": "Failed to extract content from PDF files"}), 500
    
    # Print sample documents for debugging
    if all_docs:
        print(f"Sample document 1: {all_docs[0]}")
        if len(all_docs) > 1:
            print(f"Sample document 2: {all_docs[1]}")
    
    # Create vector database
    try:
        vectordb = Chroma.from_documents(
            documents=all_docs,
            embedding=get_embeddings(),
            persist_directory=CHROMA_DIR
        )
        
        return jsonify({
            "success": "Vector DB created successfully",
            "document_count": len(all_docs),
            "retrieved document": basename_list
        }), 201
    except Exception as e:
        return jsonify({"error": f"Failed to create vector database: {str(e)}"}), 500

@vectordb_bp.route('/status', methods=['GET'])
def check_status():
    """Check if the vector database exists and is accessible"""
    try:
        if os.path.exists(CHROMA_DIR):
            # Try to initialize the DB to verify it's valid
            vectordb = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=get_embeddings(),
            )
            collection_count = len(vectordb._client.list_collections())
            
            return jsonify({
                "status": "available",
                "location": CHROMA_DIR,
                "collections": collection_count
            }), 200
        else:
            return jsonify({
                "status": "not_found",
                "location": CHROMA_DIR
            }), 404
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500