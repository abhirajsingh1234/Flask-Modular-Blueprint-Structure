import os
from flask import Blueprint, request, jsonify
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama.embeddings import OllamaEmbeddings
load_dotenv()

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
    )
llm = OllamaLLM(model="gemma3:12b")

PROMPT_TEMPLATE = """
You are an AI assistant that provides accurate answers based on provided document context.
If the answer is not in the context, reply "I don't know.".

Context: {context}

Question: {question}
Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./_chromadb")
chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/', methods=['GET'])
def home():
    return jsonify({"message": "getting started with chat"}), 200
@chat_bp.route('/', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    question = data.get("question")
    if not question:
        return jsonify({"error": "`user query` is required"}), 400

    # Initialize Chroma vector DB
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="pdf_collection"
    )

    # Create retriever and fetch relevant documents
    retriever = vectordb.as_retriever(search_kwargs={"k": 3,"filter": {"source": "Risk_Policy.pdf"}})
    docs = retriever.get_relevant_documents(question)

    # Log retrieved chunks
    for doc in docs:
        print(f"\nMetadata: {doc.metadata}")
        print(f"Content (preview): {doc.page_content[:500]}\n")

    # Set up memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    # Set up Conversational Retrieval Chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    # Query the chain
    result = qa({"question": question})

    # Process chat history
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
