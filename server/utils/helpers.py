import requests
from PIL import Image
from io import BytesIO
from datetime import timedelta
import os
import dotenv
dotenv.load_dotenv()
#helper functions
#goal:env
def get_chunks_by_topic(vectorstore, topic_query):
    all_docs = vectorstore.similarity_search("placeholder", k=len(vectorstore.docstore._dict))
    
    topic_chunks = []
    for doc in all_docs:
        if doc.metadata.get("topic", "").lower() == topic_query.lower():
            topic_chunks.append(doc.page_content)
    
    return topic_chunks


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

def format_time(seconds):# not in use
    td = str(timedelta(seconds=seconds)).split(".")[0]
    return td + ",000"  

def write_srt(subtitles, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for idx, start, end, sentence in subtitles:
            f.write(f"{idx}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{sentence}\n\n")

LIGHTHOUSE_API_KEY = os.getenv("LIGHTHOUSE_API_KEY")

def upload_to_lighthouse(video_path):
    url = "https://node.lighthouse.storage/api/v0/add"
    headers = {
        "Authorization": f"Bearer 4616e2eb.a19087940c234f68be253ffacf47d3e9",
    }
    with open(video_path, "rb") as f:
        response = requests.post(url, headers=headers, files={"file": f})
        response.raise_for_status()
        return response.json()["Hash"]