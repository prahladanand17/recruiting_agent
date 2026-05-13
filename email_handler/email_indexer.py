from datetime import datetime
import logging
import os
import json

import chromadb
from tqdm import tqdm
import voyageai


def _tqdm_disabled() -> bool:
    return os.getenv("TQDM_DISABLE", "").strip().lower() in ("1", "true", "yes")


def _email_id_from_chunk_id(chunk_id: str) -> str | None:
    """Chunk ids are '{email_id}-{index}'; recover email_id when suffix is numeric."""
    base, sep, suffix = chunk_id.rpartition("-")
    if not sep or not suffix.isdigit():
        return None
    return base


class EmailIndexer():
    def __init__(self, debug=False, log_level=logging.INFO, collection_name="emails"):
        self.debug = debug
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(collection_name)

        self.voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.voyage_model = "voyage-4-large"

    def _indexed_email_ids(self) -> set[str]:
        """
        Single paginated scan of Chroma ids (no embeddings).
        Avoids per-email get() round-trips and fixes chunk id shape ({email_id}-{i}).
        """
        indexed: set[str] = set()
        offset = 0
        page_size = 10_000
        while True:
            res = self.collection.get(
                limit=page_size,
                offset=offset,
                include=["metadatas"],
            )
            ids = res.get("ids") or []
            if not ids:
                break
            for cid in ids:
                parent = _email_id_from_chunk_id(cid)
                if parent is not None:
                    indexed.add(parent)
            if len(ids) < page_size:
                break
            offset += len(ids)
        return indexed

    def _metadata_for_email(self, email: dict) -> dict:
        raw = str(email.get("date") or "")[:26]
        try:
            ts = datetime.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S").timestamp()
        except (ValueError, TypeError):
            ts = 0.0
        return {
            "id": email["id"],
            "sender_name": email["sender_name"],
            "sender": email["sender"],
            "subject": email["subject"],
            "date": ts,
            "read": email.get("read", False),
        }

    def embed_texts(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Embed any flat list of strings in Voyage-sized batches (default 128)."""
        if batch_size is None:
            batch_size = max(1, min(int(os.getenv("VOYAGE_EMBED_BATCH_SIZE", "128")), 256))
        out: list[list[float]] = []
        batches = range(0, len(texts), batch_size)
        for i in tqdm(
            batches,
            desc="Voyage embeddings",
            unit="batch",
            disable=_tqdm_disabled() or len(texts) == 0,
        ):
            batch = texts[i : i + batch_size]
            result = self.voyage_client.embed(
                batch, model=self.voyage_model, input_type="document"
            )
            out.extend(result.embeddings)
        return out

    def index_emails_batch(self, emails: list[dict]) -> int:
        """
        Chunk many emails, one Voyage sweep over all chunk texts, one Chroma add.
        Much faster than per-email index_email for cold builds.
        """
        if not emails:
            return 0
        all_ids: list[str] = []
        all_docs: list[str] = []
        all_metas: list[dict] = []
        email_iter = tqdm(
            emails,
            desc="Chunking emails",
            unit="email",
            disable=_tqdm_disabled(),
        )
        for email in email_iter:
            chunks = self.chunk_email(email["body"])
            meta = self._metadata_for_email(email)
            eid = email["id"]
            for i, ch in enumerate(chunks):
                all_ids.append(f"{eid}-{i}")
                all_docs.append(ch["text"])
                all_metas.append(meta)
        if not all_ids:
            return 0
        self.logger.info(
            "Batch-indexing %s emails (%s chunks) — single embed pass + Chroma add",
            len(emails),
            len(all_ids),
        )
        embeddings = self.embed_texts(all_docs)
        slice_size = max(500, int(os.getenv("CHROMA_ADD_CHUNK_SLICE", "16000")))
        slice_starts = range(0, len(all_ids), slice_size)
        for start in tqdm(
            slice_starts,
            desc="Chroma write",
            unit="slice",
            disable=_tqdm_disabled() or len(all_ids) == 0,
        ):
            end = start + slice_size
            self.collection.add(
                ids=all_ids[start:end],
                documents=all_docs[start:end],
                embeddings=embeddings[start:end],
                metadatas=all_metas[start:end],
            )
        return len(emails)

    def index_email(self, email):
        """Index one email; delegates to batch path for consistent behavior."""
        self.index_emails_batch([email])

    def chunk_email(self, email_body, chunk_size=200, overlap=20):
        #Email is a json object with the relevant metadata defined in the email_handler/emails.json file
        step = chunk_size - overlap
        chunks = []
        for start in range(0, len(email_body), step):
            end = min(start + chunk_size, len(email_body))
            chunk_text = email_body[start:end]
            if not chunk_text:
                break
            chunks.append({"text": chunk_text, "start": start, "end": end})
            if end >= len(email_body):
                break
        return chunks   

    def embed_email_chunks(self, email_chunks):
        """Embed chunks from one email (list of dicts with 'text' or raw str)."""
        texts = [
            c if isinstance(c, str) else (c.get("text") or "")
            for c in email_chunks
        ]
        return self.embed_texts(texts)


    #TODO: Get emails from Gmail
    def get_emails(self, email_file="email_handler/emails.json"):
        #Temporarily load emails from a JSON file
        print(f"Loading emails from {email_file}")
        with open(email_file, "r") as f:
            emails = json.load(f)
        return emails

    def write_emails_to_chroma(self):
        emails = self.get_emails()
        already = self._indexed_email_ids()
        chunk_total = self.collection.count()
        self.logger.info(
            "Chroma: %s chunks, ~%s distinct emails already indexed",
            chunk_total,
            len(already),
        )
        to_index = [e for e in emails if e["id"] not in already]
        new_count = len(to_index)
        if to_index:
            self.index_emails_batch(to_index)
            for e in to_index:
                already.add(e["id"])
        skipped = len(emails) - new_count
        self.logger.info(
            "Indexing complete: %s new, %s skipped (already in Chroma)",
            new_count,
            skipped,
        )