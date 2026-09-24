# memory.py
import re
from pathlib import Path
from datetime import datetime

import frontmatter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import VAULT_DIR, CHROMA_DIR, OLLAMA_BASE_URL, EMBED_MODEL


class MemorySystem:
    def __init__(self):
        for folder in ["conversations", "memories", "briefings", "agent_notes"]:
            (VAULT_DIR / folder).mkdir(parents=True, exist_ok=True)

        self.embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embeddings,
            collection_name="atlas_memory",
        )
        self._index_vault()

    # ── Indexing ─────────────────────────────────────────────────────────

    def _index_vault(self):
        md_files = list(VAULT_DIR.rglob("*.md"))
        if not md_files:
            return

        existing = self.vectorstore.get()
        indexed  = {
            m.get("source", "") for m in existing["metadatas"]
        } if existing["metadatas"] else set()

        new_texts, new_metas = [], []
        for filepath in md_files:
            if str(filepath) in indexed:
                continue
            try:
                post = frontmatter.load(filepath)
                text = f"{post.metadata}\n\n{post.content}".strip()
                if text:
                    new_texts.append(text)
                    new_metas.append({
                        "source": str(filepath),
                        "type":   str(post.metadata.get("type", "note")),
                        "date":   str(post.metadata.get("date", "")),
                    })
            except Exception as e:
                print(f"[Memory] Could not index {filepath}: {e}")

        if new_texts:
            self.vectorstore.add_texts(texts=new_texts, metadatas=new_metas)
            print(f"[Memory] Indexed {len(new_texts)} new document(s).")

    # ── Retrieval ────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 4) -> str:
        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return "No relevant memories found."
        lines = ["Relevant memories:\n"]
        for doc in results:
            lines.append("---")
            lines.append(doc.page_content[:600])
            lines.append("")
        return "\n".join(lines)

    def find_note(self, query: str) -> tuple[str, str, float]:
        """
        Return (filepath, page_content, relevance_score) for the best matching note.
        Returns ("", "", 0.0) if nothing relevant found.
        """
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=1)
        if not results:
            return "", "", 0.0
        doc, score = results[0]
        return doc.metadata.get("source", ""), doc.page_content, score

    # ── Writing ──────────────────────────────────────────────────────────

    def _write_and_index(self, filepath: Path, content: str, index_text: str, meta: dict):
        """Write full content to disk but index only index_text in ChromaDB."""
        filepath.write_text(content, encoding="utf-8")
        self.vectorstore.add_texts(texts=[index_text], metadatas=[meta])

    def write_conversation(self, user_input: str, agent_response: str, summary: str = ""):
        ts = datetime.now()
        fn = ts.strftime("%Y-%m-%d_%H-%M-%S") + ".md"
        fp = VAULT_DIR / "conversations" / fn

        full_md = f"""---
date: {ts.strftime("%Y-%m-%d %H:%M:%S")}
type: conversation
---
# Conversation — {ts.strftime("%Y-%m-%d %H:%M")}

**User:** {user_input}

**Atlas:** {agent_response}
"""
        if summary:
            full_md += f"\n## Summary\n{summary}\n"

        # Index the summary if provided (cleaner), otherwise the full exchange
        index_text = summary if summary else f"User: {user_input}\nAtlas: {agent_response}"

        self._write_and_index(fp, full_md, index_text, {
            "source": str(fp),
            "type":   "conversation",
            "date":   ts.isoformat(),
        })

    def write_note(self, title: str, content: str, note_type: str = "memory"):
        ts   = datetime.now()
        safe = re.sub(r"[^\w\s\-]", "", title).strip().replace(" ", "_")
        fp   = VAULT_DIR / "agent_notes" / f"{safe}.md"
        md   = f"""---
date: {ts.strftime("%Y-%m-%d %H:%M:%S")}
type: {note_type}
title: {title}
---
# {title}

{content}
"""
        self._write_and_index(fp, md, md, {
            "source": str(fp),
            "type":   note_type,
            "date":   ts.isoformat(),
        })
        return str(fp)

    def write_briefing(self, content: str, briefing_type: str = "daily"):
        ts = datetime.now()
        fn = ts.strftime("%Y-%m-%d") + f"_{briefing_type}.md"
        fp = VAULT_DIR / "briefings" / fn
        md = f"""---
date: {ts.strftime("%Y-%m-%d %H:%M:%S")}
type: briefing
briefing_type: {briefing_type}
---
# {briefing_type.title()} Briefing — {ts.strftime("%Y-%m-%d")}

{content}
"""
        self._write_and_index(fp, md, content, {
            "source": str(fp),
            "type":   "briefing",
            "date":   ts.isoformat(),
        })

    # ── Deletion ─────────────────────────────────────────────────────────

    def delete_note(self, filepath: str) -> bool:
        """
        Delete a note from both the filesystem and the ChromaDB index.
        Returns True on success.
        """
        path = Path(filepath)

        # Remove from ChromaDB
        try:
            result = self.vectorstore._collection.get(
                where={"source": str(path)}
            )
            ids = result.get("ids", [])
            if ids:
                self.vectorstore.delete(ids=ids)
        except Exception as e:
            print(f"[Memory] ChromaDB deletion error: {e}")

        # Remove from filesystem
        if path.exists():
            path.unlink()
            print(f"[Memory] Deleted: {path.name}")
            return True

        return False
