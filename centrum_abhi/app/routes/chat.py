import os
from flask import Blueprint, request, jsonify
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama.embeddings import OllamaEmbeddings
from app.extensions import db
from app.models import ConversationHistory
from datetime import datetime
load_dotenv()

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
    )
llm = OllamaLLM(model="gemma3:12b")

PROMPT_TEMPLATE = """
You are an AI assistant that provides accurate answers **only** from the provided document context.
If the answer is not present, reply "I don't know." **(do not hallucinate).**

➤ **Write the answer directly** – no preambles like "Based on the context" and no meta commentary.
➤ **Be comprehensive**: paraphrase and include every relevant detail that appears in the context.
➤ Do **not** add information that is not explicitly in the context.

Context:
{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./_chromadb")
chat_bp = Blueprint('chat', __name__)

# Initialization flags and variables
is_initialized = False
vectordb = None
retriever = None
qa_chain = None

# def get_vectordb():
#     global vectordb
#     if vectordb is None:
#         try:
#             print("Initializing vector database...")
#             vectordb = Chroma(
#                 persist_directory=CHROMA_DIR,
#                 embedding_function=embeddings,
#             )
#             print("Vector DB initialized.")
#         except Exception as e:
#             print("Failed to initialize vector DB:", e)
#     return vectordb
# vectordb = None 
# vectordb  = get_vectordb()                   
# retriever = vectordb.as_retriever(               
#     search_type="similarity",
#     search_kwargs={"k": 3},
# )


# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True,
#     output_key="answer"
# )

# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm = llm,
#     retriever = retriever,
#     memory = memory,
#     chain_type = "stuff",
#     combine_docs_chain_kwargs = {"prompt": prompt},
#     return_source_documents = True,
# )

@chat_bp.route('/', methods=['GET'])
def home():
    return jsonify({"message": "getting started with chat"}), 200
@chat_bp.route('/', methods=['POST'])
def chat():
    global is_initialized, vectordb, retriever, qa_chain

    if not is_initialized:
        print("🔧 Initializing vector DB and LLM chain...")

        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        llm = OllamaLLM(model="gemma3:12b")

        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="pdf_collection"
        )
        print("Total docs in vector DB:", len(vectordb.get()["documents"]))

        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

        is_initialized = True
        print(" Initialization complete.")
    data        = request.get_json(force=True)
    question    = data.get("question")
    doc_filter  = data.get("retrieved document")  
    convo_id = data.get("convo_id",1) 
    user_name = data.get("user_name", "abhiraj")

    if not question:
        return jsonify({"error": "`user query` is required"}), 400


    if doc_filter:

        retriever.search_kwargs["filter"] = {"source": doc_filter}

    else:

        retriever.search_kwargs.pop("filter", None)

    result = qa_chain({"question": question})

    if convo_id: 
        conversation_record = ConversationHistory(
            convo_id=convo_id,
            human_message=question,
            ai_message=result["answer"],
            user_name=user_name,
            request_datetime=datetime.utcnow(),
            response_datetime=datetime.utcnow()
        )
        
        db.session.add(conversation_record)
        db.session.commit()
    
    chat_history = []
    if "chat_history" in result:
        for msg in result["chat_history"]:
            if hasattr(msg, 'content'):
                role = "human" if hasattr(msg, 'example') and msg.example == False else "assistant"
                chat_history.append({"role": role, "content": msg.content})

    # Return answer + metadata
    return jsonify({
        "answer": result.get("answer", ""),
        "metadata": [
            {
                "source": doc.metadata.get("source", ""),
                "topic": doc.metadata.get("topic", ""),
                "subtopic": doc.metadata.get("subtopic", ""),
                "page_no": doc.metadata.get("page_no", "")
            }
            for doc in result.get("source_documents", [])
        ],
        # "chat_history": chat_history
    }), 200
