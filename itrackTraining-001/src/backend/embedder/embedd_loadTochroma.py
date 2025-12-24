# embedd_loadTochroma.py
# Embeds documents and performs smart upserts into ChromaDB + PostgreSQL
# Uses data_hash as unique document ID (one vector per trainee/entity)
# Uses version_hash to detect content changes and update only when needed

import os
import chromadb
import psycopg2
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from indexing_chunking import create_index_jsonl


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
MAIN_FOLDER = r"C:\Users\devavaishnee.m\Desktop\itrack-chatbot\itrackTraining-001"
CHROMA_PATH = os.path.join(MAIN_FOLDER, "src", "database", "chromadb_data")
COLLECTION_NAME = "project_vectors"

BATCH_SIZE = 100


def upsert_to_chroma_and_postgres(main_folder_path: str):
    print("=" * 90)
    print("EMBEDDING & SMART UPSERT TO CHROMA + POSTGRES".center(90))
    print("=" * 90)

    # Get documents directly from indexing
    print("Retrieving processed documents...")
    json_data = create_index_jsonl(main_folder_path)

    if not json_data:
        print("No documents to process. Exiting early.")
        print("=" * 90)
        return

    print(f"Received {len(json_data):,} documents for embedding")

    # Initialize Chroma
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="hybrid_rag_db",
            user="rag_app_user",
            password="ragapp"
        )
        cur = conn.cursor()
    except Exception as e:
        raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")

    # Prepare records: use data_hash as unique ID
    incoming_records = []
    for item in json_data:
        meta = item["metadata"]
        incoming_records.append({
            "id": meta["data_hash"],              # ← Unique ID per trainee/entity
            "data_hash": meta["data_hash"],
            "version_hash": meta["version_hash"],
            "text": item["text"],
            "metadata": meta
        })

    # Get existing records to compare version_hash
    incoming_data_hashes = list({rec["data_hash"] for rec in incoming_records})  # Unique data_hashes only

    if incoming_data_hashes:
        existing_results = collection.get(
            where={"data_hash": {"$in": incoming_data_hashes}},
            include=["metadatas"]  # IDs returned automatically
        )
    else:
        existing_results = {"ids": [], "metadatas": []}

    # Map: data_hash → current version_hash in Chroma
    existing_version_map = {}
    for chromadb_id, meta in zip(existing_results.get("ids", []), existing_results.get("metadatas", [])):
        if meta and "data_hash" in meta:
            existing_version_map[meta["data_hash"]] = meta.get("version_hash")

    # Classify: only one record per data_hash should be upserted
    to_upsert = []
    new_count = updated_count = skipped_count = 0

    # Deduplicate by data_hash (keep only one per trainee — latest or any)
    seen_data_hashes = set()
    for rec in incoming_records:
        dh = rec["data_hash"]
        if dh in seen_data_hashes:
            continue  # Skip duplicates — keep first one
        seen_data_hashes.add(dh)

        current_vh = existing_version_map.get(dh)
        if current_vh is None:
            to_upsert.append(rec)
            new_count += 1
        elif current_vh != rec["version_hash"]:
            to_upsert.append(rec)
            updated_count += 1
        else:
            skipped_count += 1

    print(f"\nClassification (unique by data_hash):")
    print(f"   New records       : {new_count:,}")
    print(f"   Updated records   : {updated_count:,}")
    print(f"   Skipped (unchanged): {skipped_count:,}")

    if not to_upsert:
        print("\nNo changes detected. Knowledge base is up to date.")
        total = collection.count()
        print(f"Collection '{COLLECTION_NAME}' has {total:,} vectors")
        conn.close()
        print("=" * 90)
        return

    # Upsert in batches
    print(f"\nUpserting {len(to_upsert):,} documents (one per trainee) in batches of {BATCH_SIZE}...")

    for i in range(0, len(to_upsert), BATCH_SIZE):
        batch = to_upsert[i:i + BATCH_SIZE]
        texts = [rec["text"] for rec in batch]
        embeddings = ef(texts)

        collection.upsert(
            ids=[rec["id"] for rec in batch],           # data_hash → unique per entity
            embeddings=embeddings,
            metadatas=[rec["metadata"] for rec in batch],
            documents=texts
        )

        # PostgreSQL upsert (using data_hash as document_id)
        try:
            for rec in batch:
                cur.execute("""
                    INSERT INTO document_chunks 
                        (document_id, raw_text_chunk, data_hash, version_hash, date_ingested)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (document_id) DO UPDATE SET
                        raw_text_chunk = EXCLUDED.raw_text_chunk,
                        data_hash = EXCLUDED.data_hash,
                        version_hash = EXCLUDED.version_hash,
                        date_ingested = NOW()
                """, (
                    rec["id"],  # data_hash
                    rec["text"],
                    rec["data_hash"],
                    rec["version_hash"]
                ))
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"PostgreSQL upsert failed: {e}")

    total = collection.count()
    print(f"\nSUCCESS! Collection '{COLLECTION_NAME}' now contains {total:,} vectors")
    print("KNOWLEDGE BASE UPDATED EFFICIENTLY — ONE VECTOR PER TRAINEE!")
    print("=" * 90)

    conn.close()


# Manual testing
def main():
    upsert_to_chroma_and_postgres(MAIN_FOLDER)


if __name__ == "__main__":
    main()