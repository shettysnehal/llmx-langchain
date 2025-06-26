from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import tempfile
from duckduckgo_search import DDGS
import time
import requests
from utils.helpers import get_chunks_by_topic, try_download, is_valid_image, estimate_timings, format_time,split_into_chunks, upload_to_lighthouse
from templates.few_shots import examples
from templates.prompts import speech_prompt_template, topic_prompt_template
import pyttsx3
from langchain_unstructured import UnstructuredLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from langchain_groq import ChatGroq
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import cv2
import pytesseract
import torch
from datetime import timedelta
from moviepy import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

import re, dotenv, json

#load pdf,save the vectorstore
#run the search engine and get the chunks by topic
#generate video

dotenv.load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])



# Load models once
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=700,timeout=None,
    max_retries=2,)

engine = pyttsx3.init()
engine.setProperty('rate', 173)  # Set speech rate
voices = engine.getProperty('voices')

    # Select male voice (usually index 0 or try looping to find one)
for voice in voices:
    if 'male' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

@app.route("/api/upload-blog", methods=["POST"])
def upload_pdf():
    print("Received request to upload PDF",request)
    content_id = request.form.get("contentId")  # or pass via query or form
    if not content_id:
        return jsonify({"error": "contentId is required"}), 400

    vector_dir = os.path.join(os.getcwd(), "vectors")
    os.makedirs(vector_dir, exist_ok=True)
    vector_path = os.path.join(vector_dir, f"{content_id}.pkl")
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
        # Load document using UnstructuredLoader
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
        with open(vector_path, "wb") as f:
            pickle.dump(vectorstore, f)
            print(f" Vectorstore saved to {vector_path}")
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

@app.route("/api/generate-vid", methods=["POST"])  
def generate_video():
    
    os.makedirs("retrieved_images", exist_ok=True)
    data = request.json
    print(data)
    topic = data.get("topic")
    content_id = data.get("contentId")
    if not content_id:
        return jsonify({"error": "contentId not provided"}), 400
    if not topic:
        return jsonify({"error": "No topic provided"}), 400
    vector_path = os.path.join("vectors", f"{content_id}.pkl")
    if not os.path.exists(vector_path):
        return jsonify({"error": "Vectorstore not found for contentId"}), 404
    with open(vector_path, "rb") as f:
        vectorstore = pickle.load(f)

    print(f"Vectorstore loaded for {content_id}")
    chunks = get_chunks_by_topic(vectorstore, topic)
    if len(chunks)>20:
        chunks = chunks[:20]
    example_prompt = PromptTemplate.from_template(
    "Topic: {topic}\nChunks: {chunks}\nDescriptors: {descriptors}"
    )
    descriptor_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    
    suffix=topic_prompt_template,
    input_variables=["topic", "chunks"]
    )
    final_prompt =descriptor_prompt.format(topic=topic, chunks=chunks)
    response = llm.predict(final_prompt)
    data = json.loads(response)
    for key, value in data.items():
        print(f"{value}")
    ddgs = DDGS()
    for key, query in data.items():
        time.sleep(10)  #for rate limiting
        results = ddgs.images(
            keywords=query,
            region="wt-wt",
            safesearch="off",
            size='Large',
            type_image=None,
            layout=None,
            license_image=None,
            max_results=3,
        )

        found = False
        for r in results:
            time.sleep(1)
            if try_download(r['image'], f"retrieved_images/{key}.jpg") and is_valid_image(requests.get(r['image']).content):
                print(f" Downloaded {key}")
                found = True
                break
        if not found:
            print(f"Failed to download any valid image for {key}")

        
    speech_prompt = PromptTemplate.from_template(speech_prompt_template)

    all_results ={}

    for file in os.listdir("retrieved_images"):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  
        img_path = os.path.join("retrieved_images", file)
        img_cv = cv2.imread(img_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  #
    

        img_processeed = Image.fromarray(gray)
        extracted_text   = pytesseract.image_to_string(img_processeed)
        print(extracted_text) 
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image,"The image is a ", return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        final_prompt = speech_prompt.format(
        topic=topic,
        chunks=",".join(chunks).strip(),
        ocr_text=extracted_text.strip(),
        caption=caption.strip(),
        )
        response = llm.invoke(final_prompt)
        print(response.content)
        all_results[file] = {
            "caption": caption.strip(),
            "extracted_text": extracted_text.strip(),
            "speech": response.content.strip()
        }
    with open("results.json", "w",encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("Results saved to results.json")
    os.makedirs("audio_files", exist_ok=True)



    with open("results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    for file, data in results.items():
        speech_text = data["speech"]
        if speech_text.strip():
            engine.save_to_file(speech_text, f"audio_files/{file}.mp3")
            print(f"Audio saved for {file}")
        else:   
            print(f"No speech generated for {file}, skipping audio generation.")
    engine.runAndWait() 


    # Ensure output directory exists
    os.makedirs("final_video", exist_ok=True)


    all_video_clips = []
    caption_width = 1000

    for key, value in results.items():
        image_path = f"retrieved_images/{key}"
        audio_path = f"audio_files/{key}.mp3"
        speech = value["speech"]
        
        
        # Estimate subtitle timings
        subtitles = estimate_timings(speech,160)  # Should return list of (idx, start, end, sentence)

        # Load image and audio
        image_clip = ImageClip(image_path).resized((1280, 720))  # Resize to 1280x720
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        image_clip = image_clip.with_duration(duration)
        image_clip.audio = audio_clip.subclipped(0, duration)

        # Generate subtitle text cli
        text_clips = []
        for idx, start, end, sentence in subtitles:
            txt = TextClip(
                text=sentence,
                font_size=33,
                size=(caption_width, 100),
                method="caption",
                color='black',
            
            )
            txt = txt.with_start(start).with_duration(end - start).with_position('bottom')
            text_clips.append(txt)

        # Combine image and subtitles
        composite = CompositeVideoClip([image_clip] + text_clips)
        all_video_clips.append(composite)

    # Concatenate all clips into a single video
    final_video = concatenate_videoclips(all_video_clips)
    final_video.write_videofile("final_video/final_combined_video.mp4", fps=24)
    final_dir = os.path.join(os.getcwd(), "final_video")
    os.makedirs(final_dir, exist_ok=True)
    video_path = os.path.join(final_dir, "final_combined_video.mp4")

    try:
            # Save the video here (use moviepy or however it's generated)
            final_video.write_videofile(video_path, fps=24)

            # Upload to Lighthouse
            cid = upload_to_lighthouse(video_path)
            print(" Video uploaded to IPFS with CID:", cid)

            # Remove the video after upload
            os.remove(video_path)

            return jsonify({"ipfs": cid})
    except Exception as e:
            print(" Failed to upload video:", e)
            return jsonify({"error": "Failed to upload video", "details": str(e)}), 500
        
if __name__ == "__main__":
    app.run(debug=True)
