def get_chunks_by_topic(vectorstore, topic_query):
    all_docs = vectorstore.similarity_search("placeholder", k=len(vectorstore.docstore._dict))
    
    topic_chunks = []
    for doc in all_docs:
        if doc.metadata.get("topic", "").lower() == topic_query.lower():
            topic_chunks.append(doc.page_content)
    
    return topic_chunks

