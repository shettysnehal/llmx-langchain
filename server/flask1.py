from langchain_unstructured import UnstructuredLoader
from langchain.schema import Document
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts import PromptTemplate

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from langchain_groq import ChatGroq
import os
import dotenv
import re
from duckduckgo_search import DDGS
import requests
from PIL import Image
from io import BytesIO
import time
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
import cv2
import pytesseract
import torch
from datetime import timedelta
from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips






# Initialize
model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=700,
    timeout=None,
    max_retries=2,
) 
dotenv.load_dotenv()

topic = #take it from api flask #


# Load the PDF document
loader_local = UnstructuredLoader(
    file_path="evs.pdf",
    strategy="hi_res",#use fast for faster loading
    mode="elements",  # use "elements" to get structured data
)
docs_local = []   
section_data =[]
section ="" 
for doc in loader_local.lazy_load():
    docs_local.append(doc)

for docs in docs_local:
    if docs.metadata.get("category") == "Title":
        
        section_data.append(section)
        section =""
        section+= docs.page_content + "\n"
        
    else:
        section += docs.page_content + "\n"





# Suppose section_data is a list of section texts
all_chunks = []
raw_chunks =[]

for i, section in enumerate(section_data):
    if len(section) > 0:
        chunks = text_splitter.split_text(section)
        print(f"Section {i} has {len(chunks)} chunks")
        for j, chunk in enumerate(chunks):
            # Optional: Add metadata like section number
            all_chunks.append(Document(
                page_content=chunk,
                metadata={"section_id": i, "chunk_id": j}
            ))
            raw_chunks.append(chunk)

# Create the vector index
embeddings = model.embed_documents(raw_chunks)



#cluster embeddings

num_topics = 10
kmeans = KMeans(n_clusters=num_topics, random_state=42)
labels = kmeans.fit_predict(embeddings)




cluster_topic_titles = {}
for cluster_id in set(labels):
    rep_idx = list(labels).index(cluster_id)
    rep_chunk = raw_chunks[rep_idx]

    # Ask LLM to name this topic
    # Updated prompt
    prompt = (
        f"Give a very short and clear title for the following topic content.\n"
        f"Just return the title. No explanations, no quotes, no alternatives, no extra text.\n\n"
        f"{rep_chunk}"
    )
    raw_title = llm.invoke(prompt).content.strip()
    clean_title = re.sub(r'^["“”‘’\'*]*|["“”‘’\'*.:]*$', '', raw_title)  # trim quotes, punctuation
    clean_title = re.sub(r'^(Topic Title|Title)\s*[:\-]\s*', '', clean_title, flags=re.IGNORECASE)
    clean_title = clean_title.split("\n")[0].strip()
    cluster_topic_titles[cluster_id] = clean_title
    
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
vectorstore = FAISS.from_texts(
    texts=[chunk["text"] for chunk in labeled_chunks],
    embedding=model,
    metadatas=[chunk["metadata"] for chunk in labeled_chunks]
)

#retriever
retriever = vectorstore.as_retriever(search_kwargs=dict(k=5))
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

 # or input("Enter topic: ")
chunks = get_chunks_by_topic(vectorstore, topic)
if len(chunks)>20:

    chunks = chunks[:20]
print(len(chunks))
# Create individual prompt template for each example
example_prompt = PromptTemplate.from_template(
    "Topic: {topic}\nChunks: {chunks}\nDescriptors: {descriptors}"
)


descriptor_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    
    suffix=template,
    input_variables=["topic", "chunks"]
)
final_prompt =descriptor_prompt.format(topic=topic, chunks=chunks)
response = llm.predict(final_prompt)
import json
data = json.loads(response)
for key, value in data.items():
    print(f"{value}")


os.makedirs("retrieved_images", exist_ok=True)

def try_download(image_url, filepath):
    try:
        res = requests.get(image_url, timeout=5)
        if res.status_code == 200 and 'image' in res.headers.get('Content-Type', ''):
            with open(filepath, 'wb') as f:
                f.write(res.content)
            return True
    except:
        pass
    return False
def is_valid_image(image_bytes, min_width=400, min_height=300, min_size_kb=30):
    try:
        img = Image.open(BytesIO(image_bytes))
        width, height = img.size
        file_size_kb = len(image_bytes) / 1024
        return width >= min_width and height >= min_height and file_size_kb >= min_size_kb
    except:
        return False
ddgs = DDGS()
for key, query in data.items():
    time.sleep(10)  # Be kind to the API and avoid rate limiting
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
            print(f"✅ Downloaded {key}")
            found = True
            break
    if not found:
        print(f"❌ Failed to download any valid image for {key}")

speech_prompt = PromptTemplate.from_template(speech_prompt_template)
 # or input("Enter topic: ")
chunks = get_chunks_by_topic(vectorstore, topic)
if len(chunks)>20:
    chunks = chunks[:20]
print(",".join(chunks).strip())
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
import json
import pyttsx3
import os
os.makedirs("audio_files", exist_ok=True)

engine = pyttsx3.init()
engine.setProperty('rate', 173)  # Set speech rate
voices = engine.getProperty('voices')

# Select male voice (usually index 0 or try looping to find one)
for voice in voices:
    if 'male' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break


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
    
   


# === STEP 1: Utilities for Subtitle Generation ===

def split_into_chunks(text, start, end, chunk_size=5):
    words = text.split()
    total_words = len(words)
    total_chunks = (total_words + chunk_size - 1) // chunk_size
    duration_per_chunk = (end - start) / total_chunks

    chunks = []
    for i in range(total_chunks):
        chunk_text = ' '.join(words[i*chunk_size : (i+1)*chunk_size])
        chunk_start = start + i * duration_per_chunk
        chunk_end = chunk_start + duration_per_chunk
        chunks.append((None, chunk_start, chunk_end, chunk_text))
    return chunks

def estimate_timings(speech_text, wpm):
    sentences = [s.strip() for s in speech_text.split('.') if s.strip()]
    all_chunks = []
    start = 0.0
    idx = 1
    for sentence in sentences:
        word_count = len(sentence.split())
        duration = word_count / (wpm / 60.0)
        end = start + duration
        chunks = split_into_chunks(sentence, start, end, chunk_size=5)
        for chunk in chunks:
            all_chunks.append((idx, chunk[1], chunk[2], chunk[3]))
            idx += 1
        start = end
    return all_chunks

def format_time(seconds):
    td = str(timedelta(seconds=seconds)).split(".")[0]
    return td + ",000"

def write_srt(subtitles, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for idx, start, end, sentence in subtitles:
            f.write(f"{idx}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{sentence}\n\n")
#without captions


# Ensure output directory exists
os.makedirs("final_video", exist_ok=True)

# Load results
with open("results.json", "r") as f:
    results = json.load(f)

all_video_clips = []

for key, value in results.items():
    image_path = f"retrieved_images/{key}"
    audio_path = f"audio_files/{key}.mp3"

    # Load and resize image, load audio
    image_clip = ImageClip(image_path).resized((1280, 720))
    audio_clip = AudioFileClip(audio_path)

    # Set duration and audio
    duration = audio_clip.duration
    image_clip = image_clip.with_duration(duration)
    image_clip.audio = audio_clip.subclipped(0, duration)

    # Append to the video clips list
    all_video_clips.append(image_clip)

# Concatenate all clips into a single video
final_video = concatenate_videoclips(all_video_clips)
final_video.write_videofile("final_video/final_combined_video_without_caption.mp4", fps=24)
###dont save the video give as response in the api 

    
    
    
        

    # Path to your image file   

    