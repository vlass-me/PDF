import tiktoken
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS # 내부 메모리 저장이라 나중에 퀴즈낼 때 사용 못함 
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
from sklearn.cluster import KMeans
import openai
import pinecone
from langchain.vectorstores.pinecone import Pinecone

OPENAI_API_KEY = "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"
PINECONE_API_KEY = "6757054f-e22e-4481-83a3-84b5ed0b0db8"
PINECONE_ENVIRONMENT = "us-east1-gcp"
INDEX_NAME = "foo"

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = ""
    for page in pages:
        text += page.page_content
    text = text.replace('\t', ' ')
    return text

def load_text(text_path):
    loader = TextLoader(text_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs

def generate_summary(text):
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        max_tokens=3000,
        model='gpt-4',
        request_timeout=120
    )

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    docs = text_splitter.create_documents(chunks)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectors = embeddings.embed_documents([x.page_content for x in docs])

    num_clusters = 1

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

    closest_indices = []

    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)

    selected_docs = [docs[doc] for doc in selected_indices]

    summaries = []

    for i, doc in enumerate(selected_docs):
        prompt = f"{doc.page_content}\n\nTl;dr"
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.1,
            max_tokens=1000,
            top_p=1.0,
        )
        summary = response.choices[0].text
        summaries.append(summary)

    return summaries

def main():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    
    pdf_path = "CTC.pdf"
    text = load_pdf(pdf_path)
    summaries = generate_summary(text)
    
    for i, summary in enumerate(summaries):
        print(f"Summary #{i+1}:")
        print(summary)
        print()

    
    # Load text from a file
    #text_path = "../../../state_of_the_union.txt"
    #docs = load_text(text_path)

    # Create Pinecone index
    #docsearch = Pinecone.from_documents(docs, OpenAIEmbeddings())

    #query = "What did the president say about Ketanji Brown Jackson"
    #docs = docsearch.similarity_search(query)
    #print(docs)

if __name__ == '__main__':
    main()
