from langchain_community.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader(file_path="nlp.pdf",mode="elements", strategy="fast" )
docs_local = []
for doc in loader.lazy_load():
    docs_local.append(doc)
print("Loaded documents:", len(docs_local))