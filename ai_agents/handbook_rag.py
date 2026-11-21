"""
Academic Policy RAG Engine: ChromaDB Vector Store for University Handbook
Indexes university policy regulations and provides offline semantic retrieval.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HandbookRAG")

AI_AGENTS_DIR = Path(__file__).resolve().parent
HANDBOOK_FILE = AI_AGENTS_DIR / "university_handbook.txt"
CHROMA_DIR = AI_AGENTS_DIR / "chroma_db"
COLLECTION_NAME = "university_handbook_policies"


def get_embedding_function():
    """
    Returns open-source embedding function.
    Defaults to Chroma's built-in all-MiniLM-L6-v2 ONNX embeddings.
    """
    try:
        return embedding_functions.DefaultEmbeddingFunction()
    except Exception as e:
        logger.warning(f"DefaultEmbeddingFunction warning: {e}. Falling back to default.")
        return None


def parse_handbook_sections(file_path: Path = HANDBOOK_FILE) -> List[Dict[str, Any]]:
    """
    Parses the university handbook into structured semantic chunks with rich metadata.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Handbook file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into sections based on delimiter or headings
    chunks = []
    current_section = "General"
    current_topic = "General"

    lines = content.split("\n")
    current_chunk = []
    chunk_title = "Apex University Academic Policy"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Section 1:"):
            if current_chunk:
                chunks.append({
                    "title": chunk_title,
                    "section": current_section,
                    "topic": current_topic,
                    "text": "\n".join(current_chunk).strip(),
                })
                current_chunk = []
            current_section = "Section 1: Attendance Policies and Condonation"
            current_topic = "Attendance & Condonation"
            chunk_title = stripped
        elif stripped.startswith("Section 2:"):
            if current_chunk:
                chunks.append({
                    "title": chunk_title,
                    "section": current_section,
                    "topic": current_topic,
                    "text": "\n".join(current_chunk).strip(),
                })
                current_chunk = []
            current_section = "Section 2: Continuous Assessment & Remedial Class Directives"
            current_topic = "Internal Marks & Remedials"
            chunk_title = stripped
        elif stripped.startswith("Section 3:"):
            if current_chunk:
                chunks.append({
                    "title": chunk_title,
                    "section": current_section,
                    "topic": current_topic,
                    "text": "\n".join(current_chunk).strip(),
                })
                current_chunk = []
            current_section = "Section 3: Merit & Need-Based Financial Aid Waivers"
            current_topic = "Scholarship & Financial Aid"
            chunk_title = stripped
        elif stripped.startswith("Section 4:"):
            if current_chunk:
                chunks.append({
                    "title": chunk_title,
                    "section": current_section,
                    "topic": current_topic,
                    "text": "\n".join(current_chunk).strip(),
                })
                current_chunk = []
            current_section = "Section 4: Academic Probation and Early Intervention Protocol"
            current_topic = "Academic Probation & Advising"
            chunk_title = stripped
        elif stripped.startswith(("1.1", "1.2", "1.3", "2.1", "2.2", "2.3", "3.1", "3.2", "3.3", "4.1", "4.2", "4.3")):
            if current_chunk:
                chunks.append({
                    "title": chunk_title,
                    "section": current_section,
                    "topic": current_topic,
                    "text": "\n".join(current_chunk).strip(),
                })
                current_chunk = []
            chunk_title = stripped
            current_chunk.append(line)
        else:
            if not stripped.startswith("====") and not stripped.startswith("----"):
                current_chunk.append(line)

    if current_chunk:
        chunks.append({
            "title": chunk_title,
            "section": current_section,
            "topic": current_topic,
            "text": "\n".join(current_chunk).strip(),
        })

    # Filter out empty or whitespace-only chunks
    valid_chunks = [c for c in chunks if len(c["text"].strip()) > 30]
    return valid_chunks


def get_chroma_collection(force_reindex: bool = False):
    """
    Initializes or retrieves the persistent ChromaDB collection for the handbook.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    emb_fn = get_embedding_function()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"description": "Apex University Academic Regulations and Student Policies"},
    )

    count = collection.count()
    if count == 0 or force_reindex:
        logger.info(f"Indexing handbook policies into ChromaDB at {CHROMA_DIR}...")
        if force_reindex and count > 0:
            client.delete_collection(name=COLLECTION_NAME)
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=emb_fn,
            )

        chunks = parse_handbook_sections()
        ids = [f"policy_rule_{i+1:02d}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [{
            "title": c["title"],
            "section": c["section"],
            "topic": c["topic"],
        } for c in chunks]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info(f"Successfully indexed {len(chunks)} policy clauses into ChromaDB.")

    return collection


def query_handbook(query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    Queries the vector database for the most semantically relevant handbook policies.
    """
    try:
        collection = get_chroma_collection()
        results = collection.query(query_texts=[query_text], n_results=n_results)

        retrieved = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for i in range(len(docs)):
                retrieved.append({
                    "title": metas[i].get("title", "University Policy"),
                    "section": metas[i].get("section", "Academic Code"),
                    "topic": metas[i].get("topic", "General"),
                    "text": docs[i],
                    "relevance_distance": round(float(distances[i]), 4) if distances else 0.0,
                })
        return retrieved
    except Exception as e:
        logger.error(f"Error querying handbook vector store: {e}")
        # Fallback keyword matching directly from handbook
        return fallback_keyword_search(query_text, n_results)


def fallback_keyword_search(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """Keyword search fallback if vector store encounter issues."""
    chunks = parse_handbook_sections()
    query_lower = query.lower()
    keywords = [w for w in query_lower.split() if len(w) > 3]

    scored = []
    for c in chunks:
        score = sum(1 for kw in keywords if kw in c["text"].lower() or kw in c["title"].lower())
        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "title": c["title"],
            "section": c["section"],
            "topic": c["topic"],
            "text": c["text"],
            "relevance_distance": 0.5,
        }
        for _, c in scored[:n_results]
    ]


def format_policies_for_prompt(policies: List[Dict[str, Any]]) -> str:
    """Formats retrieved policy clauses as structured text for LLM prompt context."""
    if not policies:
        return "No specific handbook policies retrieved."

    formatted = []
    for idx, p in enumerate(policies, 1):
        formatted.append(f"[{idx}] {p['title']} ({p['section']}):\n{p['text']}\n")
    return "\n".join(formatted)


if __name__ == "__main__":
    print("Initializing & Indexing University Handbook RAG Engine...")
    col = get_chroma_collection(force_reindex=True)
    print(f"Total Policy Rules in ChromaDB: {col.count()}")

    test_queries = [
        "chronic low attendance below 70%",
        "failing internal marks remedial tutoring",
        "scholarship retention and GPA financial aid grace period",
    ]

    for q in test_queries:
        print(f"\n--- Query: '{q}' ---")
        matches = query_handbook(q, n_results=2)
        for m in matches:
            print(f"  * {m['title']} (Topic: {m['topic']})")
            print(f"    Excerpt: {m['text'][:140]}...")

