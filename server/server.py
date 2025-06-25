from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile

from utils.helpers import get_chunks_by_topic
from templates.few_shots import examples
from templates.prompts import speech_prompt_template, topic_prompt_template

from langchain_unstructured import UnstructuredLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from langchain_groq import ChatGroq
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

import re, dotenv, json

dotenv.load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Load models once
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=700,timeout=None,
    max_retries=2,)

@app.route("/api/upload-blog", methods=["POST"])
def upload_pdf():
    print(request.files["pdf"])
    if 'pdf' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['pdf']
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400

    # Save file temporarily
    upload_folder = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    save_path = os.path.join(upload_folder, file.filename)
    file.save(save_path)
    print(f"Saved uploaded PDF to {save_path}")
    try:
        print(f"Processing file: {save_path}")
        # Load document
        loader = UnstructuredLoader(file_path=save_path, strategy="hi_res")
        docs_local = []    
        print("Loading document...")
        for doc in loader.lazy_load():
            docs_local.append(doc)
        print("loaded success")       # Parse into sections
        section_data, section = [], ""
        for doc in docs_local:
            if doc.metadata.get("category") == "Title":
                section_data.append(section)
                section = doc.page_content + "\n"
            else:
                section += doc.page_content + "\n"
        

        # Print section titles
        print("Section Titles:")
        first_page_docs = [doc for doc in docs_local if doc.metadata.get("page_number") == 3]

        for doc in first_page_docs:
            print(doc.page_content)


        # Chunk sections
        all_chunks, raw_chunks = [], []
        for i, section in enumerate(section_data):
            if len(section.strip()) > 0:
                chunks = text_splitter.split_text(section)
                print(f"Section {i} has {len(chunks)} chunks")
                for j, chunk in enumerate(chunks):
                    all_chunks.append(Document(
                        page_content=chunk,
                        metadata={"section_id": i, "chunk_id": j}
                    ))
                    raw_chunks.append(chunk)

        # Embedding + Clustering
        embeddings = embedding_model.embed_documents(raw_chunks)
        num_topics = 6
        kmeans = KMeans(n_clusters=num_topics, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # Title for each cluster
        cluster_topic_titles = {}
        for cluster_id in set(labels):
            rep_idx = list(labels).index(cluster_id)
            rep_chunk = raw_chunks[rep_idx]
            prompt = (
                f"Give a very short and clear title for the following topic content.\n"
                f"Just return the title. No explanations, no quotes, no alternatives, no extra text.\n\n"
                f"some of the chunks could be about authors,publications,government details,production details **ignore such chunks**.\n"
                f"{rep_chunk}"
            )
            raw_title = llm.invoke(prompt).content.strip()
            clean_title = re.sub(r'^["“”‘’\'*]*|["“”‘’\'*.:]*$', '', raw_title)
            clean_title = re.sub(r'^(Topic Title|Title)\s*[:\-]\s*', '', clean_title, flags=re.IGNORECASE)
            clean_title = clean_title.split("\n")[0].strip()
            cluster_topic_titles[cluster_id] = clean_title

        # Label chunks with topic
        labeled_chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_meta = {
                "section_id": all_chunks[i].metadata["section_id"],
                "chunk_id": all_chunks[i].metadata["chunk_id"],
                "cluster_id": int(labels[i]),
                "topic": cluster_topic_titles[labels[i]]
            }
            labeled_chunks.append({
                "text": chunk_text,
                "embedding": embeddings[i],
                "metadata": chunk_meta
            })

        # Create FAISS index
        vectorstore = FAISS.from_texts(
            texts=[chunk["text"] for chunk in labeled_chunks],
            embedding=embedding_model,
            metadatas=[chunk["metadata"] for chunk in labeled_chunks]
        )
        # === Done! You can now use vectorstore.as_retriever() ===
        """  retriever = vectorstore.as_retriever(search_kwargs=dict(k=5)) """

        # Get all stored documents from FAISS
        all_docs = vectorstore.similarity_search("placeholder", k=len(vectorstore.docstore._dict))

        # Extract and print all unique topics
        topics = set()
        for doc in all_docs:
            topic = doc.metadata.get("topic")
            if topic:
                topics.add(topic)

        print("Unique Topics:")
        for topic in sorted(topics):
            print("-", topic)

        # Return topics as JSON
        return jsonify(sorted(list(topics)))



    finally:
        
        try:
            os.remove(save_path)
            print(f"Deleted uploaded file: {save_path}")
        except Exception as e:
            print(f"Error deleting uploaded file: {e}")
if __name__ == "__main__":
    app.run(debug=True)
