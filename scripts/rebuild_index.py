"""
Rebuild FAISS index from data/docs/ knowledge base

Usage:
    python scripts/rebuild_index.py
    python scripts/rebuild_index.py --download-crud
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env API key
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v)

from config import DEFAULT_CONFIG
from rag.embedder import EmbeddingManager
from rag.indexer import IndexManager
from rag import get_chunker
from langchain_core.documents import Document

DOCS_DIR = Path("data/docs")
INDEX_DIR = Path("data/index")


def load_docs_from_dir(docs_dir: Path) -> list:
    """Load all .txt files from data/docs/"""
    if not docs_dir.exists():
        print(f"[ERROR] Directory not found: {docs_dir}")
        return []

    txt_files = sorted(docs_dir.glob("*.txt"))
    if not txt_files:
        print(f"[ERROR] No .txt files in: {docs_dir}")
        return []

    docs = []
    for fp in txt_files:
        if fp.name == "_index.txt":
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                content = f.read().strip()
            if len(content) < 20:
                continue
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": str(fp),
                    "source_type": "text",
                    "file_name": fp.name,
                },
            ))
        except Exception as e:
            print(f"  [WARN] Skipped {fp.name}: {e}")

    return docs


def download_crud_docs():
    """Download CRUD-RAG 80K news documents to data/docs/crud_80000/"""
    import urllib.request
    import json

    target_dir = DOCS_DIR / "crud_80000"
    target_dir.mkdir(parents=True, exist_ok=True)

    api_url = "https://api.github.com/repos/IAAR-Shanghai/CRUD_RAG/contents/data/80000_docs"
    print(f"[DOWNLOAD] Fetching file list: {api_url}")

    try:
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            file_list = json.loads(resp.read().decode())
    except Exception as e:
        print(f"[ERROR] Failed to fetch file list: {e}")
        return False

    total_files = len(file_list)
    total_size = sum(f["size"] for f in file_list)
    print(f"    {total_files} files, {total_size/1024/1024:.1f} MB total")
    print()

    total_lines = 0
    for i, file_info in enumerate(file_list):
        fname = file_info["name"]
        fpath = target_dir / fname
        if fpath.exists():
            print(f"  [{i+1}/{total_files}] [OK] Already exists: {fname}")
            with open(fpath, "r", encoding="utf-8") as f:
                total_lines += sum(1 for _ in f)
            continue

        url = file_info["download_url"]
        print(f"  [{i+1}/{total_files}] [DOWNLOAD] {fname} ({file_info['size']/1024:.0f}KB)...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, fpath)
            with open(fpath, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
            total_lines += lines
            print(f"[OK] ({lines} docs)")
        except Exception as e:
            print(f"[FAIL] {e}")

    print(f"\n[OK] Download complete: {total_lines} news documents")
    return True


def build_index(docs, config):
    """Build FAISS index from documents"""
    if not docs:
        print("[ERROR] No documents to index")
        return

    print(f"\n[DOCS] Total: {len(docs)} documents")
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"   Total chars: {total_chars:,}")

    # Chunk
    print(f"\n[CHUNK] Splitting...")
    t0 = time.time()
    chunker_cls = get_chunker()
    chunker = chunker_cls()
    chunks = chunker.chunk_all(docs)
    chunk_time = time.time() - t0
    print(f"   {len(docs)} docs -> {len(chunks)} chunks ({chunk_time:.1f}s)")
    chunk_chars = sum(len(c.page_content) for c in chunks)
    print(f"   Chunk total chars: {chunk_chars:,}")

    # Embedding + FAISS
    print(f"\n[EMBED] Vectorizing + building FAISS index...")
    t0 = time.time()
    embedder = EmbeddingManager(config.embedding_model)
    indexer = IndexManager(embedder.get_embeddings(), str(INDEX_DIR))
    indexer.build_index(chunks)
    indexer.save_index()
    index_time = time.time() - t0

    index_path = INDEX_DIR / "index.faiss"
    print(f"\n[OK] Index built!")
    print(f"   Time: {index_time:.1f}s")
    print(f"   Index: {index_path}")
    print(f"   Vectors: {indexer.index.ntotal}")
    print(f"   Dimension: {indexer.index.d}")

    # Estimate tokens
    avg_chars_per_token = 1.8  # Chinese chars per token approx
    est_tokens = int(chunk_chars / avg_chars_per_token)
    print(f"\n[TOKENS] Estimated embedding tokens: ~{est_tokens:,}")
    print(f"   (text-embedding-v2: ~${est_tokens/1000*0.0007:.4f} beyond free tier)")

    return indexer


def main():
    parser = argparse.ArgumentParser(description="Rebuild RAG knowledge base index")
    parser.add_argument("--download-crud", action="store_true", help="Download CRUD 80K docs first")
    parser.add_argument("--use-crud-80000", action="store_true", help="Use CRUD 80K docs (must be downloaded)")
    args = parser.parse_args()

    t_start = time.time()

    if args.download_crud:
        download_crud_docs()

    docs = []
    if args.use_crud_80000:
        crud_dir = DOCS_DIR / "crud_80000"
        if crud_dir.exists():
            print(f"\n[LOAD] Loading CRUD 80K documents...")
            for f in sorted(crud_dir.iterdir()):
                if f.is_file():
                    with open(f, encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                docs.append(Document(
                                    page_content=line,
                                    metadata={
                                        "source": f.name,
                                        "source_type": "text",
                                        "file_name": f.name,
                                    },
                                ))
            print(f"   Total: {len(docs)} documents")
        else:
            print(f"[ERROR] CRUD 80K dir not found. Run --download-crud first.")
            return
    else:
        docs = load_docs_from_dir(DOCS_DIR)

    build_index(docs, DEFAULT_CONFIG)

    elapsed = time.time() - t_start
    print(f"\n[TIME] Total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
