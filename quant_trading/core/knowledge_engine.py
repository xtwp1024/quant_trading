# -*- coding: utf-8 -*-
"""
Titan Knowledge Engine (TKE)
Internal semantic memory system for Titan V13, providing RAG capabilities.
"""

import logging
import hashlib
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger("Titan.KnowledgeEngine")


class TitanKnowledgeEngine:
    """
    Lightweight RAG Engine for Titan V13.
    """
    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        if persist_dir:
            self.persist_dir = persist_dir
        else:
            # Default to relative data directory within project
            self.persist_dir = Path(__file__).parent.parent / "data" / "chroma_db"

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Initializing Titan Knowledge Engine @ {self.persist_dir}")
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(name="titan_knowledge")
            logger.info(f"Connection successful. Items: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Batch add documents. Format: [{"id":str, "content":str, "metadata":dict}]
        """
        if not documents:
            return

        valid_docs: List[str] = []
        valid_ids: List[str] = []
        valid_metas: List[Dict[str, Any]] = []

        for doc in documents:
            content = doc.get("content")
            if not content:
                continue

            doc_id = doc.get("id") or hashlib.md5(content.encode()).hexdigest()
            metadata = doc.get("metadata", {})
            metadata["timestamp"] = time.time()

            # Ensure metadata values are compatible with Chroma
            cleaned_meta = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}

            valid_docs.append(content)
            valid_ids.append(doc_id)
            valid_metas.append(cleaned_meta)

        if valid_docs:
            self.collection.upsert(
                documents=valid_docs,
                metadatas=valid_metas,
                ids=valid_ids
            )
            logger.debug(f"Added {len(valid_docs)} items to knowledge base.")

    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            memories: List[Dict[str, Any]] = []
            if results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    memories.append({
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "score": results['distances'][0][i] if results.get('distances') else 0.5
                    })
            return memories
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = TitanKnowledgeEngine()
    engine.add_documents([{"content": "Titan V13 is an event-driven system.", "metadata": {"source": "manual"}}])
    print(engine.recall("What is Titan?"))
