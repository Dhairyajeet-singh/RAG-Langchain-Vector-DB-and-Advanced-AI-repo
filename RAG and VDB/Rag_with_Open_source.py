import os 
import re 
import json 
import uuid 
import argparse
from pathlib import Path
from typing import Optional
from openai import OpenAI
from pinecone import Pinecone , ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME      = "rag-index"
EMBED_MODEL     = "text-embedding-3-small"
EMBED_DIM       = 1536
CHAT_MODEL      = "gpt-4o"
CLOUD           = "aws"
REGION          = "us-east-1"

CHUNK_SIZE      = 512   
CHUNK_OVERLAP   = 64
TOP_K           = 5

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

def load_documents(file_path: str) -> list[dict]:
    """
    Load documents from a text file and split them into chunks.
    supports .txt, .md, few .pdf
    this solution is basic and will support basic files, advanced text file will be covered in the langchain and llamaindex part

    """
    docs = []
    paths = [Path(file_path)] if Path(file_path).is_file() else list(Path(file_path).rglob("*"))

    for path in paths:
        if path.suffix in {".txt", ".md"}:
            docs.append({"text": path.read_text(errors="replace"), "source": str(path)})

        elif path.suffix == ".pdf":
            try:
                from pdfminer.high_level import extract_text
                text = extract_text(str(path))
                docs.append({"text": text, "source": str(path)})
            except ImportError:
                print(f"pdfminer is required to process PDF files. Skipping {path}.")
                continue
        print(f"[load] {len(docs)} documents loaded from {file_path}")
        return docs
    
    """
    here we completed the document loading part, 
    we will load the documents and split them into chunks,
      we will also keep track of the source of each chunk for later use in retrieval 
      and attribution
    """

    #part 2 chunking the documents

def chunk_text(text: str, size: int=CHUNK_SIZE, overlap: int=CHUNK_OVERLAP)-> list[str]:
    """
    Split text into chunks of specified size with overlap.
    """
    text = re.sub(r"\s+"," ", text).strip()
    chunks, start = [] , 0
    while start < len(text):
        end = min(start+size, len(text))

        if end<len(text):
            boundary  = text.rfind(". ", start, end)
            if boundary != -1:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]

def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Chunk documents and maintain source attribution.
    """
    chunks= []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "source": f"{doc['source']} (chunk {i})"
            })
    print(f"[chunk] {len(chunks)} chunks created from {len(docs)} documents")
    return chunks

#embedding

def embed_texts(text: list[str], batch_size: int = 100 ) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using OpenAI API.
    """
    embeddings = []
    for i in range(0, len(text), batch_size):
        batch = text[i:i+batch_size]
        response = openai_client.embeddings.create(input=batch, model = EMBED_MODEL)
        embeddings.extend([r.embedding for r in response.data])
    return embeddings

#vector db store

def get_or_create_index():
    """
    Get or create Pinecone index for storing embeddings.
    """
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        print(f"[pinecone] Created index '{INDEX_NAME}'")
    else:
        print(f"[pinecone] Index '{INDEX_NAME}' already exists")
    return pc.Index(INDEX_NAME)

def upsert_chunks(index, chunks: list[dict]):
    """
    Upsert chunk embeddings into Pinecone index.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": chunk["id"],
            "values": emb,
            "metadata": {
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": i,       # <— FIXED HERE
            },
        })
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])

    print(f"[pinecone] upserted {len(vectors)} vector(s)")

def retrieve(index, query:str , top_k : int = TOP_K, filter: Optional[dict] = None) -> list[dict]:
    """
    Retrieve relevant chunks from Pinecone index based on query.
    """
    query_emb = embed_texts([query])[0]
    result = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True,
        filter=filter
    )
    return [
        {
            "text": match.metadata["text"],
            "source": match.metadata["source"],
            "score": match.score
        }
        for match in result.matches
    ]

SYSTEM_PROMPT = (
    "You are an AI assistant answering a user question using the retrieved context below."
    "Write a detailed, well-structured answer with a minimum of 10-12 sentences. "
    "Do not copy from the context; instead, synthesize and explain the ideas clearly."
    "Expand on concepts, provide examples, and include relevant implications or trends."
    "If the context discusses multiple themes, combine them into a single coherent explanation."
    "Always cite which source(s) you used."
)

def generate(query: str, retrieved: list[dict]) -> str:
    """Build a prompt from retrieved chunks and call the chat model."""
    context_parts = []
    for i, chunk in enumerate(retrieved, 1):
        context_parts.append(f"[{i}] (source: {chunk['source']})\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content

class RAG:

    def __init__(self):
        self.index = get_or_create_index()

    def ingest(self, file_path: str):
        docs = load_documents(file_path)
        chunks = chunk_documents(docs)
        upsert_chunks(self.index, chunks)

    def query(self, query:str, top_k: int= TOP_K, filter: Optional[dict] = None, verbose: bool = False) -> str:
        retrieved = retrieve(self.index, query, top_k=top_k, filter=filter)
        if verbose:
            for i, r in enumerate(retrieved, 1):
                print(f"[{i}] {r['source']} (score: {r['score']:.4f})\n{r['text']}\n")
                print(f"    {r['text'][:200]}...\n")

        return generate(query, retrieved)
    

if __name__ == "__main__":
     
    parser = argparse.ArgumentParser(description="RAG with OpenAI and Pinecone")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_p = sub.add_parser("ingest", help="Ingest documents from a file or directory"
                              )
    ingest_p.add_argument("file_path", type=str, help="Path to file or directory to ingest")

    query_p = sub.add_parser("query", help="Query the RAG system")
    query_p.add_argument("query", type=str, help="Query string")
    query_p.add_argument("--top_k", type=int, default=TOP_K, help="Number of top chunks to retrieve")
    query_p.add_argument("--verbose", action="store_true", help="Print retrieved chunks")

    args = parser.parse_args()
    rag = RAG()

    if args.command == "ingest":
        rag.ingest(args.file_path)
        print(f"Documents ingested from {args.file_path}")
    elif args.command == "query":
        response = rag.query(args.query, top_k=args.top_k, verbose=args.verbose)
        print(response)







"""
Documents (txt / md / pdf)
        │
        ▼  load_documents()
   Raw text per file
        │
        ▼  chunk_documents()
  Overlapping text chunks (512 chars, 64 overlap)
        │
        ▼  embed_texts()          [OpenAI text-embedding-3-small]
  Float vectors (1536-dim)
        │
        ▼  upsert_chunks()        [Pinecone serverless]
   Vector index
        │
  ── query time ──────────────────────────────
        │
  User question
        │
        ▼  embed_texts()
  Query vector
        │
        ▼  retrieve()             [cosine similarity top-k]
  Top-k chunks + metadata
        │
        ▼  generate()             [GPT-4o]
  Grounded answer with citations
"""