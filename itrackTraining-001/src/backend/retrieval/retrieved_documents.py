import os
import json
from typing import List, Dict
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Load env
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
load_dotenv(os.path.join(project_root, '.env'))

COLLECTION_NAME = "project_vectors"
current_script_dir = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(current_script_dir, "..", "..", "database", "chromadb_data")

class MiniLMEmbedder(HuggingFaceEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
            **kwargs
        )

embedder = MiniLMEmbedder()

def get_vectorstore():
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=embedder
        )
        print(f" Chroma loaded: {vectorstore._collection.count()} documents")
        return vectorstore
    except Exception as e:
        print(f" Error loading Chroma: {e}")
        return None

_bm25_retriever = None

def get_bm25_retriever() -> BM25Retriever:
    global _bm25_retriever
    if _bm25_retriever is None:
        print(" Building BM25 index...")
        vectorstore = get_vectorstore()
        if vectorstore:
            all_data = vectorstore._collection.get(include=["documents", "metadatas"])
            docs = all_data["documents"]
            metas = all_data["metadatas"]
            
            texts = []
            bm25_metadatas = []
            for i, doc in enumerate(docs):
                # Use searchable_text for BM25 (keyword matching)
                searchable_text = metas[i].get("searchable_text", doc)
                texts.append(searchable_text)  
                doc_id = metas[i].get("document_id", f"doc_{i}")
                bm25_metadatas.append({"document_id": doc_id})
            
            documents = [Document(page_content=t, metadata=m) for t, m in zip(texts, bm25_metadatas)]
            _bm25_retriever = BM25Retriever.from_documents(documents)
            print(f" BM25 ready ({len(docs)} docs)")
    return _bm25_retriever

def retrieve_bm25(query: str, n_results: int = 10) -> List[Dict]:
    retriever = get_bm25_retriever()
    if not retriever:
        return []
    docs = retriever.invoke(query)[:n_results]
    results = []
    for rank, doc in enumerate(docs, 1):
        doc_id = doc.metadata.get('document_id')
        bm25_score = 1.0 / rank
        
        # Get original JSON from metadata or page_content
        try:
            # Try metadata first (new format), then page_content (old format)
            json_str = doc.metadata.get("full_json") or doc.page_content
            json_doc = json.loads(json_str)
        except (json.JSONDecodeError, KeyError):
            json_doc = {"error": "Invalid JSON", "raw_text": doc.page_content[:200]}
        
        results.append({
            'id': doc_id,
            'data': json_doc,
            'bm25_score': round(bm25_score, 3),
            'bm25_rank': rank
        })
    return results

def retrieve_hybrid(
    user_query: str,
    n_results: int = 3,
    semantic_weight: float = 0.7,
    bm25_weight: float = 0.3,
    min_threshold: float = 0.5 
) -> List[Dict]:
    print(f"\n{'='*80}")
    print(f"QUERY: '{user_query}'")
    
    vectorstore = get_vectorstore()
    if not vectorstore:
        print(" No vectorstore")
        return []
    
    # SEMANTIC (FIXED COSINE)
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(
            query=user_query, k=n_results * 4
        )
        
        semantic_docs = []
        for doc, score in docs_with_scores:
            
            cosine_sim = round(1 - (score / 2.0), 3) 
            
            try:
                json_doc = json.loads(doc.page_content)  # JSON in page_content
            except:
                json_doc = {"error": "Invalid JSON"}
            
            doc.metadata['score'] = cosine_sim
            semantic_docs.append((doc, cosine_sim, json_doc))
        
        best_sem = max([s[1] for s in semantic_docs]) if semantic_docs else 0
        print(f" Semantic: {len(semantic_docs)} docs (best cosine: {best_sem:.3f})")
        
    except Exception as e:
        print(f" Semantic error: {e}")
        semantic_docs = []
    
    # BM25
    bm25_results = retrieve_bm25(user_query, n_results * 4)
    best_bm25 = bm25_results[0]['bm25_score'] if bm25_results else 0
    print(f" BM25: {len(bm25_results)} docs (best: {best_bm25:.3f})")
    
    # RRF FUSION (70/30) - WORKING PERFECTLY
    score_map = {}
    
    # Semantic RRF
    for rank, (doc, cosine_sim, json_doc) in enumerate(semantic_docs, 1):
        doc_id = doc.metadata.get('document_id')
        if doc_id:
            rrf_sem = 60 / (60 + rank - 1)
            weighted_sem = rrf_sem * semantic_weight
            
            score_map[doc_id] = score_map.get(doc_id, {
                'score': 0, 'cosine_similarity': 0, 'data': {}, 'source': ''
            })
            score_map[doc_id]['score'] += weighted_sem
            score_map[doc_id]['cosine_similarity'] = cosine_sim
            score_map[doc_id]['data'] = json_doc
            score_map[doc_id]['source'] = 'both' if doc_id in [b['id'] for b in bm25_results] else 'semantic'
    
    # BM25 RRF  
    for bm25_doc in bm25_results:
        doc_id = bm25_doc['id']
        if doc_id:
            weighted_bm25 = bm25_doc['bm25_score'] * bm25_weight
            
            if doc_id not in score_map:
                score_map[doc_id] = {
                    'score': weighted_bm25,
                    'cosine_similarity': 0.0,
                    'data': bm25_doc['data'],
                    'source': 'bm25'
                }
            else:
                score_map[doc_id]['score'] += weighted_bm25
                score_map[doc_id]['source'] = 'both'
    
    # FINAL RESULTS
    final_results = []
    for doc_id, info in sorted(score_map.items(), key=lambda x: x[1]['score'], reverse=True):
        if info['score'] >= min_threshold:
            final_results.append({
                'id': doc_id,
                'data': info['data'],
                'relevance_score': round(info['score'], 3),
                'cosine_similarity': round(info['cosine_similarity'], 3),
                'source': info['source']
            })
            if len(final_results) >= n_results:
                break
    
    print(f" FINAL: {len(final_results)} docs ≥ {min_threshold}")
    return final_results
def format_llm_context(results: List[Dict]) -> str:
    if not results:
        return "No highly relevant documents found."
    
    parts = []
    for r in results:
        # Pretty print key fields
        data = r['data']
        preview = f"Type: {data.get('record_type', 'unknown')}\n"
        if data.get('record_type') == 'TRAINER_ASSESSMENT':
            preview += f"Trainer: {data.get('Trainer_Name', 'N/A')}\n"
            preview += f"Python: {data.get('Technical_Aggregate_Python', 'N/A')}\n"
        elif data.get('record_type') == 'TRAINEE_LOG':
            preview += f"Trainee: {data.get('Trainee_Name', 'N/A')}\n"
            preview += f"PTO: {data.get('PTO', 0)}\n"
        
        parts.append(f"""
ID: {r['id']} 
Source: {r['source']} | Relevance: {r['relevance_score']:.3f} | Cosine: {r['cosine_similarity']:.3f}

{preview}
Full Data:
{json.dumps(r['data'], indent=2)}
---
""")
    return "\n".join(parts)

if __name__ == "__main__":
    print(" FIXED HYBRID RAG - JSON CORRECTED")
    print(f"DB Path: {CHROMA_PATH}")
    
    queries = [
        "Eve Chen Python assessment"
    ]
    
    for q in queries:
        results = retrieve_hybrid(q, n_results=3, min_threshold=0.5)  # Lower threshold for demo
        context = format_llm_context(results)
        print(f"\n {q}")
        print(context)
        print("=" * 100)