#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated tagging script for Obsidian vaults using KeyBERT and LLMs.
"""

import os
import argparse
import logging
import sqlite3
import json
import re
import sys
import signal
from datetime import datetime
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional, Set, Any
import logging.handlers

# Third-party libraries
import spacy
import frontmatter
from ruamel.yaml import YAML
from ruamel.yaml.scanner import ScannerError
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from openai import OpenAI, APIError, RateLimitError
import unidecode
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# --- Constants ---
DEFAULT_CHUNK_SIZE_TOKENS = 1500 # Approximation, use chars as primary limit
DEFAULT_CHUNK_SIZE_CHARS = 4000
DEFAULT_MAX_TAGS_PER_DOC = 12
DEFAULT_DB_FILE = "tags.db"
DEFAULT_LOG_FILE = "tagger.log"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL_SBERT = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_EMBEDDING_MODEL_KEYBERT = 'sentence-transformers/all-MiniLM-L6-v2' # Can be same or different
DEFAULT_SPACY_MODEL = 'it_core_news_lg'
DEFAULT_SYNONYMS_FILE = "synonyms.json"

# Default prices per 1000 tokens (USD) - Overridable via CLI
DEFAULT_PRICE_TABLE = {
    "gpt-4o-mini": {"prompt": 0.60 / 1000, "completion": 2.4 / 1000}, # As of 2024-05, check current pricing
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015}
}

# LLM Prompt Structure
LLM_SYSTEM_PROMPT = "Sei un assistente che propone tag coerenti per note Obsidian in formato Markdown. Concentrati sui i temi principali, concetti chiave, entità, argomenti e categorie rilevanti, entità menzionate nel testo."
LLM_USER_PROMPT_TEMPLATE = """Analizza il seguente contenuto della nota e proponi fino a 10 tag rilevanti in inglese o italiano. 
I tag devono essere parole significative o concetti caratterizzanti il testo.
Preferisci termini singolari ai plurali.
Usa il minuscolo per standardizzare i tag
Includi sia tag generali che specifici per facilitare diversi livelli di ricerca
Non usare più di 2 parole per ogni tag
I tag devono essere in minuscolo, usare trattini per separare parole (formato kebab-case) e rappresentare concetti specifici.

Contenuto della nota:
---
{chunk_text}
---

Proposte iniziali da KeyBERT (usale come ispirazione ma non limitarti ad esse):
{keybert_tags}

Restituisci SOLO un array JSON di stringhe contenente i tag proposti. 
Esempio:["tag1", "tag2-composto", "tag3", "tag4"]
Non aggiungere commenti, spiegazioni o testo introduttivo. Assicurati che l'output sia JSON valido."""

# --- Global Variables ---
interrupted = False
db_client_global = None # Used for signal handler cleanup
log_handler_global = None # Used for signal handler flush

# --- Helper Functions ---

def split_text(text: str, max_chars: int) -> List[str]:
    """
    Splits text into chunks respecting paragraph boundaries primarily,
    and falling back to sentence boundaries if paragraphs are too long.
    Aims to stay under max_chars.
    """
    chunks = []
    current_chunk = ""
    # Split by paragraphs first
    paragraphs = [p for p in text.split('\n\n') if p.strip()]

    for paragraph in paragraphs:
        # If a single paragraph exceeds the limit, try splitting by sentences
        if len(paragraph) > max_chars:
            # Simple sentence split (may not be perfect for all cases)
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_chars:
                    current_chunk += (" " + sentence).strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    # Handle very long sentences gracefully
                    current_chunk = sentence[:max_chars] if len(sentence) > max_chars else sentence
            # Add the last part of the split paragraph if needed
            if current_chunk and current_chunk not in chunks:
                 # Avoid adding if it was just the last sentence that fit
                 if not chunks or chunks[-1] != current_chunk:
                     chunks.append(current_chunk)
                     current_chunk = "" # Reset for next paragraph


        # If adding the current paragraph doesn't exceed the limit
        elif len(current_chunk) + len(paragraph) + 1 <= max_chars:
            current_chunk += ("\n\n" + paragraph).strip()
        # If it exceeds, finalize the current chunk and start a new one
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph

    # Add the last remaining chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Sanity check chunk sizes (mostly for long sentences/paragraphs)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            # Force split if still too long
            for i in range(0, len(chunk), max_chars):
                final_chunks.append(chunk[i:i + max_chars])
        elif chunk:
             final_chunks.append(chunk)

    return final_chunks

def setup_logging(log_level_str: str, log_file: str) -> None:
    """Configures logging to console and a rotating file."""
    global log_handler_global
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s | %(levelname)-5s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    # File Handler (Rotating)
    # Rotate log file if it exceeds 10MB, keep 3 backups
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=3, encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    log_handler_global = file_handler # Store for flushing on exit

    # Root Logger Configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Silence noisy libraries if needed (optional)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("keybert").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("hdbscan").setLevel(logging.WARNING)


def load_env(env_path: Optional[str] = None) -> None:
    """Loads environment variables from .env file."""
    load_dotenv(dotenv_path=env_path)
    if not os.getenv("OPENAI_API_KEY"):
        logging.warning("OPENAI_API_KEY not found in environment variables or .env file.")
        # Depending on strictness, could raise an error here instead


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Automated tagging for Obsidian vaults using OpenAI or Ollama.")
    parser.add_argument("--vault", required=True, type=str, help="Path to the Obsidian vault directory.")
    parser.add_argument("--db", default=DEFAULT_DB_FILE, type=str, help="Path to the SQLite database file.")
    parser.add_argument("--chunk-size-chars", default=DEFAULT_CHUNK_SIZE_CHARS, type=int, help="Maximum character size for text chunks.")
    # --- LLM Provider Selection ---
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "ollama"], help="LLM provider to use.")

    # --- OpenAI Specific Arguments ---
    parser.add_argument("--openai-model", default=DEFAULT_LLM_MODEL, type=str, help="OpenAI model name (used if --llm-provider=openai).")
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"), type=str, help="OpenAI API key (required if --llm-provider=openai; overrides .env).")
    parser.add_argument("--price-prompt", type=float, help="Price per 1000 prompt tokens (USD) for OpenAI. Overrides default.")
    parser.add_argument("--price-completion", type=float, help="Price per 1000 completion tokens (USD) for OpenAI. Overrides default.")

    # --- Ollama Specific Arguments ---
    parser.add_argument("--ollama-url", default="http://localhost:11434", type=str, help="Base URL for the Ollama API (used if --llm-provider=ollama).")
    parser.add_argument("--ollama-model", type=str, default=None, help="Ollama model name (e.g., 'llama3', 'mistral'; required if --llm-provider=ollama).")

    # --- Common Arguments ---
    parser.add_argument("--max-tags-per-doc", default=DEFAULT_MAX_TAGS_PER_DOC, type=int, help="Maximum number of canonical tags to write back to a document.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level.")
    parser.add_argument("--resume", dest='resume', action='store_true', help="Resume processing, skip already processed files.")
    parser.add_argument("--no-resume", dest='resume', action='store_false', help="Start processing from scratch (default).")
    parser.set_defaults(resume=False)
    parser.add_argument("--dry-run", action='store_true', help="Run without making changes to files or database (except logging progress).")
    parser.add_argument("--reset-db", action='store_true', help="Drop and recreate database tables before starting.")
    parser.add_argument("--synonyms", default=DEFAULT_SYNONYMS_FILE, type=str, help="Path to the JSON file containing synonym mappings.")
    parser.add_argument("--env-file", type=str, default=None, help="Path to the .env file.")

    args = parser.parse_args()

    # --- Argument Validation ---
    if not Path(args.vault).is_dir():
        print(f"Error: Vault path '{args.vault}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Load OpenAI API key if provided via arg (for OpenAI provider)
    if args.llm_provider == 'openai':
        if args.openai_api_key:
            os.environ['OPENAI_API_KEY'] = args.openai_api_key
        # Check if key is available (after arg parse and .env load)
        # Do the actual check in main() after load_env() is called
    elif args.llm_provider == 'ollama':
        if not args.ollama_model:
            print("Error: --ollama-model is required when --llm-provider is 'ollama'.", file=sys.stderr)
            sys.exit(1)
        # Add /v1 to ollama url if not present (common requirement for OpenAI compatibility)
        if not args.ollama_url.endswith('/v1'):
             if args.ollama_url.endswith('/'):
                 args.ollama_url += 'v1'
             else:
                 args.ollama_url += '/v1'
             logging.info(f"Appended '/v1' to Ollama URL: {args.ollama_url}")


    return args
def signal_handler(sig, frame):
    """Handles graceful shutdown on interruption."""
    global interrupted, db_client_global, log_handler_global
    if not interrupted: # Prevent multiple prints if Ctrl+C is pressed repeatedly
        print("\nInterruption detected. Shutting down gracefully...", file=sys.stderr)
        logging.warning("Interrupt signal received. Starting graceful shutdown.")
        interrupted = True
        if db_client_global:
            print("Closing database connection...", file=sys.stderr)
            logging.info("Closing database connection.")
            db_client_global.close()
        if log_handler_global:
             print("Flushing logs...", file=sys.stderr)
             log_handler_global.flush()
             log_handler_global.close()
    # Exit after cleanup
    sys.exit(1)


# --- Classes ---

class CostTracker:
    """Tracks API call costs."""
    def __init__(self, price_table: Dict[str, Dict[str, float]]):
        self.price_table = price_table
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self.providers_used = set() # Keep track of which providers were used
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_call(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int, doc_path: Optional[str] = None):
        """Records an API call and calculates its cost if applicable."""
        self.call_count += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.providers_used.add(provider)
        cost = 0.0 # Default cost is 0 (for Ollama or unknown models)

        if provider == 'openai':
            if model not in self.price_table:
                self.logger.warning(f"OpenAI model '{model}' not found in price table. Using 0 cost for this call.")
            else:
                prices = self.price_table[model]
                cost = (prompt_tokens * prices.get('prompt', 0.0) +
                        completion_tokens * prices.get('completion', 0.0)) / 1000.0
                self.total_cost += cost

        log_prefix = f"{Path(doc_path).name} | " if doc_path else ""
        # Log cost only if it's non-zero (or always display $0.00000?) Let's show conditionally
        cost_str = f" | cost={cost:.5f}$" if cost > 0 else ""
        self.logger.info(
            f"{log_prefix}API Call {self.call_count}: provider={provider} | model={model} | "
            f"prompt_tok={prompt_tokens} | compl_tok={completion_tokens}{cost_str}"
        )
        return cost

    def report(self) -> None:
        """Logs the final cost summary."""
        provider_str = ", ".join(sorted(list(self.providers_used)))
        cost_summary = f"Estimated total cost (OpenAI only): {self.total_cost:.4f}$" if 'openai' in self.providers_used else "Estimated total cost: $0.00 (Ollama used)"

        summary = (
            f"Processing complete. LLM Provider(s) Used: {provider_str} | "
            f"Total LLM calls: {self.call_count} | "
            f"Total prompt tokens: {self.total_prompt_tokens} | "
            f"Total completion tokens: {self.total_completion_tokens} | "
            f"{cost_summary}"
        )
        self.logger.info(summary)
        print(summary) # Also print to console
        
        
class DBClient:
    """Handles all database interactions."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            # isolation_level=None enables autocommit mode initially,
            # but we will manage transactions manually per document.
            self.conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row # Access columns by name
            self.cursor = self.conn.cursor()
            self.logger.info(f"Connected to database: {db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {e}", exc_info=True)
            raise

    def _execute(self, query: str, params: tuple = ()) -> None:
        try:
            self.cursor.execute(query, params)
        except sqlite3.Error as e:
            self.logger.error(f"Database query failed: {query} | Params: {params} | Error: {e}", exc_info=True)
            raise

    def _fetchone(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            self.logger.error(f"Database fetchone failed: {query} | Params: {params} | Error: {e}", exc_info=True)
            raise
        return None

    def _fetchall(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            self.logger.error(f"Database fetchall failed: {query} | Params: {params} | Error: {e}", exc_info=True)
            raise
        return []


    def ensure_schema(self) -> None:
        """Creates database tables if they don't exist."""
        self.logger.info("Ensuring database schema exists.")
        try:
            self._execute("""
                CREATE TABLE IF NOT EXISTS doc (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    title TEXT,
                    mtime TIMESTAMP NOT NULL
                )
            """)
            self._execute("""
                CREATE TABLE IF NOT EXISTS tag (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL COLLATE NOCASE
                )
            """)
            self._execute("""
                CREATE TABLE IF NOT EXISTS tag_alias (
                    alias TEXT PRIMARY KEY COLLATE NOCASE,
                    tag_id INTEGER NOT NULL,
                    FOREIGN KEY (tag_id) REFERENCES tag(id) ON DELETE CASCADE
                )
            """)
            self._execute("""
                CREATE TABLE IF NOT EXISTS doc_tag (
                    doc_id INTEGER NOT NULL,
                    tag_id INTEGER NOT NULL,
                    PRIMARY KEY (doc_id, tag_id),
                    FOREIGN KEY (doc_id) REFERENCES doc(id) ON DELETE CASCADE,
                    FOREIGN KEY (tag_id) REFERENCES tag(id) ON DELETE CASCADE
                )
            """)
            self._execute("""
                CREATE TABLE IF NOT EXISTS progress (
                    doc_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('started', 'done', 'error')),
                    updated TIMESTAMP NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES doc(id) ON DELETE CASCADE
                )
            """)
            self.conn.commit() # Commit schema changes
            self.logger.info("Database schema verified/created.")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create database schema: {e}", exc_info=True)
            raise

    def reset_db(self) -> None:
        """Drops all known tables and recreates the schema."""
        self.logger.warning("Resetting database requested.")
        tables = ['progress', 'doc_tag', 'tag_alias', 'tag', 'doc']
        for table in tables:
            try:
                self._execute(f"DROP TABLE IF EXISTS {table}")
                self.logger.info(f"Dropped table {table}.")
            except sqlite3.Error as e:
                 self.logger.error(f"Failed to drop table {table}: {e}", exc_info=True)
                 # Continue trying to drop other tables
        self.conn.commit()
        self.ensure_schema()
        self.logger.warning("Database reset complete.")

    def get_doc_status(self, path: str, mtime: float) -> Optional[str]:
        """Checks the status of a document based on path and modification time."""
        query = """
            SELECT p.status
            FROM doc d
            JOIN progress p ON d.id = p.doc_id
            WHERE d.path = ? AND d.mtime = ?
        """
        row = self._fetchone(query, (path, datetime.fromtimestamp(mtime).isoformat()))
        return row['status'] if row else None

    def mark_status(self, doc_id: int, status: str) -> None:
        """Marks the processing status of a document."""
        now = datetime.now()
        now_iso = now.isoformat()
        query = """
            INSERT INTO progress (doc_id, status, updated)
            VALUES (?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                status = excluded.status,
                updated = excluded.updated
        """
        self._execute(query, (doc_id, status, now_iso))

    def add_or_update_doc(self, path: str, title: Optional[str], mtime: float) -> int:
        """Adds or updates a document record, returning its ID."""
        now = datetime.fromtimestamp(mtime)
        now_iso = now.isoformat()
        query_insert = """
            INSERT INTO doc (path, title, mtime) VALUES (?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                title = excluded.title,
                mtime = excluded.mtime
            RETURNING id
        """
        # For SQLite versions < 3.35 that don't support RETURNING with ON CONFLICT
        query_select = "SELECT id FROM doc WHERE path = ?"

        try:
            self.cursor.execute(query_insert, (path, title, now_iso))
            row = self.cursor.fetchone()
            if row:
                 doc_id = row['id']
            else: # Fallback for older SQLite
                row = self._fetchone(query_select, (path,))
                if not row:
                     raise sqlite3.Error("Failed to retrieve doc_id after insert/update")
                doc_id = row['id']

             # Clean old tags and progress if mtime changed (implies re-processing)
            self._execute("DELETE FROM doc_tag WHERE doc_id = ?", (doc_id,))
            # Progress is handled by mark_status called later

            return doc_id

        except sqlite3.Error as e:
             self.logger.error(f"Failed to add/update doc: {path} | Error: {e}", exc_info=True)
             raise


    def get_or_create_tag(self, tag_name: str) -> int:
        """Gets the ID of an existing tag or creates a new one."""
        # Check alias first
        row_alias = self._fetchone("SELECT tag_id FROM tag_alias WHERE alias = ?", (tag_name,))
        if row_alias:
            return row_alias['tag_id']

        # Check canonical tag table
        row_tag = self._fetchone("SELECT id FROM tag WHERE name = ?", (tag_name,))
        if row_tag:
            return row_tag['id']
        else:
            # Create new canonical tag
            try:
                self._execute("INSERT INTO tag (name) VALUES (?)", (tag_name,))
                tag_id = self.cursor.lastrowid
                if not tag_id: # Should not happen with AUTOINCREMENT
                    row_check = self._fetchone("SELECT id FROM tag WHERE name = ?", (tag_name,))
                    if not row_check: raise sqlite3.Error("Failed to retrieve tag_id after insert")
                    tag_id = row_check['id']
                # Add self-alias
                self.add_tag_alias(tag_name, tag_id)
                self.logger.debug(f"Created new tag '{tag_name}' with id {tag_id}")
                return tag_id
            except sqlite3.IntegrityError: # Handle rare race condition if another process inserted concurrently
                 row_tag = self._fetchone("SELECT id FROM tag WHERE name = ?", (tag_name,))
                 if row_tag: return row_tag['id']
                 else: raise # Re-raise if still not found


    def add_tag_alias(self, alias: str, tag_id: int) -> None:
        """Adds an alias pointing to a canonical tag ID."""
        try:
            self._execute("INSERT OR REPLACE INTO tag_alias (alias, tag_id) VALUES (?, ?)", (alias, tag_id))
        except sqlite3.Error as e:
             self.logger.error(f"Failed to add tag alias: {alias} -> {tag_id} | Error: {e}", exc_info=True)
             # Don't raise, maybe just log

    def link_doc_tag(self, doc_id: int, tag_id: int) -> None:
        """Links a document to a tag."""
        try:
            self._execute("INSERT OR IGNORE INTO doc_tag (doc_id, tag_id) VALUES (?, ?)", (doc_id, tag_id))
        except sqlite3.IntegrityError:
            self.logger.warning(f"Attempted to link doc {doc_id} to non-existent tag {tag_id} or duplicate link.", exc_info=True)
        except sqlite3.Error as e:
             self.logger.error(f"Failed to link doc {doc_id} to tag {tag_id}: {e}", exc_info=True)

    def get_doc_tags(self, doc_id: int) -> List[str]:
        """Retrieves all canonical tags associated with a document."""
        query = """
            SELECT t.name
            FROM tag t
            JOIN doc_tag dt ON t.id = dt.tag_id
            WHERE dt.doc_id = ?
            ORDER BY t.name
        """
        rows = self._fetchall(query, (doc_id,))
        return [row['name'] for row in rows]

    def get_all_aliases(self) -> Dict[str, int]:
        """Retrieves all alias->tag_id mappings."""
        rows = self._fetchall("SELECT alias, tag_id FROM tag_alias")
        return {row['alias']: row['tag_id'] for row in rows}

    def get_tag_name_by_id(self, tag_id: int) -> Optional[str]:
         """Retrieves the canonical tag name for a given ID."""
         row = self._fetchone("SELECT name FROM tag WHERE id = ?", (tag_id,))
         return row['name'] if row else None


    def commit(self) -> None:
        """Commits the current transaction."""
        if self.conn:
            try:
                self.conn.commit()
            except sqlite3.Error as e:
                 self.logger.error(f"Database commit failed: {e}", exc_info=True)

    def close(self) -> None:
        """Closes the database connection."""
        if self.conn:
            try:
                # Ensure final commit before closing, in case something was pending
                self.conn.commit()
                self.conn.close()
                self.logger.info("Database connection closed.")
            except sqlite3.Error as e:
                 self.logger.error(f"Error closing database connection: {e}", exc_info=True)


class VaultWalker:
    """Walks the vault directory and yields Markdown files."""
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path).resolve()
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.vault_path.is_dir():
            raise ValueError(f"Vault path '{vault_path}' is not a valid directory.")

    def walk(self) -> List[Path]:
        """Yields Path objects for all Markdown files, skipping .obsidian folder."""
        self.logger.info(f"Scanning vault: {self.vault_path}")
        md_files = []
        # Use rglob for simplicity, handles symlinks by default if target exists
        for item in self.vault_path.rglob('*'):
             # Check if item is within .obsidian hidden folder
             if '.obsidian' in item.parts:
                 continue

             # Process only files with .md extension
             if item.is_file() and item.suffix.lower() == '.md':
                 # Follow symlinks (though rglob often does this implicitly)
                 if item.is_symlink():
                     target = item.resolve()
                     # Check if symlink target is valid and is a file
                     if target.is_file() and target.suffix.lower() == '.md':
                         md_files.append(target)
                         self.logger.debug(f"Found symlinked file: {item} -> {target}")
                     else:
                         self.logger.debug(f"Skipping broken or non-md symlink: {item}")
                 else:
                     md_files.append(item)

        self.logger.info(f"Found {len(md_files)} Markdown files.")
        return md_files



class TagExtractor:
    """Extracts candidate tags using KeyBERT and an LLM."""
    def __init__(self,
                 openai_client: OpenAI, # Client is now generic (OpenAI SDK obj)
                 cost_tracker: CostTracker,
                 llm_provider: str,     # Added
                 model_name: str,       # Changed from llm_model
                 max_tags_per_chunk: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.keybert_model = KeyBERT(model=DEFAULT_EMBEDDING_MODEL_KEYBERT)
        self.llm_client = openai_client # Renamed for clarity
        self.cost_tracker = cost_tracker
        self.llm_provider = llm_provider # Store provider
        self.model_name = model_name     # Store the actual model name
        self.max_tags_per_chunk = max_tags_per_chunk
        self.yaml_delim = re.compile(r'^---\s*$', re.MULTILINE)
        self.code_block_delim = re.compile(r'```.*?```', re.DOTALL)
        self.html_tag = re.compile(r'<[^>]+>')
        self.whitespace_norm = re.compile(r'\s+')


    def clean_content(self, content: str) -> str:
        """Removes YAML front matter, code blocks, and HTML tags."""
        # Remove YAML front matter (if present)
        parts = self.yaml_delim.split(content, maxsplit=2)
        if len(parts) == 3 and parts[0] == '': # YAML detected
             text_content = parts[2]
        else: # No YAML or malformed
             text_content = content

        # Remove code blocks
        text_content = self.code_block_delim.sub('', text_content)
        # Remove HTML tags
        text_content = self.html_tag.sub('', text_content)
        # Normalize whitespace
        text_content = self.whitespace_norm.sub(' ', text_content).strip()
        return text_content

    def _extract_keybert_tags(self, text: str) -> List[str]:
        """Extracts keywords using KeyBERT."""
        try:
            # nr_candidates=20: Consider more candidates internally
            # keyphrase_ngram_range: Extract single words up to trigrams
            # stop_words='italian': Use basic Italian stop words
            # top_n=10: Return the top 10 relevant keyphrases
            keywords = self.keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='italian',
                use_mmr=True, # Use Maximal Marginal Relevance for diversity
                diversity=0.7,
                top_n=15, # Get slightly more to feed into LLM
                nr_candidates=20
            )
            # keywords is potentially list of tuples (phrase, score) or just phrases
            if keywords and isinstance(keywords[0], tuple):
                 return [kw[0] for kw in keywords]
            elif keywords:
                 return keywords # Assuming list of strings if not tuples
            else:
                 return []
        except Exception as e:
            self.logger.error(f"KeyBERT extraction failed: {e}", exc_info=True)
            return []



    def _extract_llm_tags(self, chunk_text: str, keybert_tags: List[str], doc_path: str) -> List[str]:
        """Uses the configured LLM provider API to extract tags."""
        user_prompt = LLM_USER_PROMPT_TEMPLATE.format(
            chunk_text=chunk_text[:8000], # Truncate just in case
            keybert_tags=", ".join(keybert_tags)
        )

        request_params = {
            "model": self.model_name, # Use the stored model name
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 250,
            "top_p": 1.0,
            "n": 1,
            "stop": None
        }

        # Add response_format only for OpenAI, as Ollama support varies
        if self.llm_provider == 'openai':
            request_params["response_format"] = {"type": "json_object"}
        else:
            request_params["response_format"] = {"type": "json_object"}
            
            
        try:
            response = self.llm_client.chat.completions.create(**request_params)

            # Ensure usage data is present (might be None for some Ollama setups/errors)
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0

            # Pass provider to cost tracker
            self.cost_tracker.add_call(
                provider=self.llm_provider,
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                doc_path=doc_path
            )

            content = response.choices[0].message.content.strip() if response.choices else ""
            if not content:
                 self.logger.warning(f"{Path(doc_path).name} | LLM returned empty content.")
                 return []

            # --- JSON Parsing Logic ---
            try:
                tags = json.loads(content)
                # ... (rest of JSON parsing logic) ...

            except json.JSONDecodeError:
                self.logger.error(f"{Path(doc_path).name} | LLM response was not valid JSON: {content}", exc_info=False) # exc_info=False to avoid logging full trace for common issue
                # ... (fallback regex logic) ...

            # Limit number of tags per chunk if needed
            return extracted_tags[:self.max_tags_per_chunk] # Ensure extracted_tags is defined in all paths


        except APIError as e: # General API errors (rate limits, auth for OpenAI, server errors)
             # Check for specific connection errors common with local Ollama
             if self.llm_provider == 'ollama' and isinstance(e.cause, ConnectionRefusedError):
                 self.logger.error(f"Connection to Ollama server refused at {self.llm_client.base_url}. Is Ollama running? Error: {e}", exc_info=False)
                 # Optional: Raise exception or sys.exit() if Ollama connection fails repeatedly? For now, just return empty.
                 return []
             elif self.llm_provider == 'ollama' and isinstance(e.cause, requests.exceptions.ConnectionError): # If using requests internally
                  self.logger.error(f"Could not connect to Ollama server at {self.llm_client.base_url}. Is Ollama running? Error: {e}", exc_info=False)
                  return []

             self.logger.error(f"API error during LLM call ({self.llm_provider}): {e}", exc_info=True)
             # Basic retry for rate limit might be added here for OpenAI
             if isinstance(e, RateLimitError) and self.llm_provider == 'openai':
                 self.logger.warning("Rate limit exceeded. Waiting 10 seconds and retrying once...")
                 time.sleep(10)
                 # Be careful with recursion here, only retry once simply
                 # return self._extract_llm_tags(chunk_text, keybert_tags, doc_path) # Simple single retry
             return []
        except ImportError as e: # Handle potential underlying library issues if requests isn't found etc.
              if 'requests' in str(e) and self.llm_provider == 'ollama':
                   self.logger.error("The 'requests' library might be required for Ollama connection via OpenAI client. Please install it: pip install requests", exc_info=False)
                   sys.exit(1) # Exit if dependency missing
              else:
                  self.logger.error(f"Import error during LLM call: {e}", exc_info=True)
                  return []

        except Exception as e: # Catch other unexpected errors
            self.logger.error(f"Unexpected error during LLM ({self.llm_provider}) API call: {e}", exc_info=True)
            return []
    def extract(self, full_content: str, doc_path: str, chunk_size_chars: int) -> List[str]:
        """Cleans content, chunks it, and extracts tags using KeyBERT and LLM."""
        cleaned_content = self.clean_content(full_content)
        if not cleaned_content:
            self.logger.info(f"{Path(doc_path).name} | No content after cleaning, skipping tag extraction.")
            return []

        chunks = split_text(cleaned_content, chunk_size_chars)
        all_raw_tags = set() # Use set to store unique raw tags from LLM

        self.logger.debug(f"{Path(doc_path).name} | Processing {len(chunks)} chunks.")

        for i, chunk in enumerate(chunks):
            if interrupted: return [] # Check for interruption

            self.logger.debug(f"{Path(doc_path).name} | Chunk {i+1}/{len(chunks)} | Size: {len(chunk)} chars")
            keybert_tags = self._extract_keybert_tags(chunk)
            self.logger.debug(f"{Path(doc_path).name} | Chunk {i+1} | KeyBERT proposals: {keybert_tags}")

            llm_tags = self._extract_llm_tags(chunk, keybert_tags, doc_path)
            self.logger.debug(f"{Path(doc_path).name} | Chunk {i+1} | LLM proposals: {llm_tags}")

            for tag in llm_tags:
                # Basic validation: kebab-case, lowercase, ALLOWING ITALIAN ACCENTS
                if re.match(r'^[a-z0-9àèéìòù]+(-[a-z0-9àèéìòù]+)*$', tag): # <--- CORRECTED REGEX
                    all_raw_tags.add(tag)
                else:
                    self.logger.warning(f"{Path(doc_path).name} | Skipping invalid tag format from LLM: '{tag}'")


        self.logger.info(f"{Path(doc_path).name} | Extracted {len(all_raw_tags)} unique raw tags from LLM across all chunks.")
        return list(all_raw_tags)


class TagNormaliser:
    """Normalises tags using lemmatisation, synonyms, and clustering."""
    def __init__(self, db_client: DBClient, synonyms_path: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_client = db_client
        self.synonyms = self._load_synonyms(synonyms_path)

        # Lazy load models on first use
        self._nlp = None
        self._sbert_model = None
        self._italian_stopwords = None

        # Clustering parameters
        self.hdbscan_min_cluster_size = 2 # Minimum size to form a cluster (2 means pairs)
        self.hdbscan_min_samples = 1      # How conservative HDBSCAN is (lower means more clusters)
        self.similarity_threshold = 0.85  # Cosine similarity threshold to consider tags similar


    def _load_nlp(self):
        if self._nlp is None:
            self.logger.info(f"Loading spaCy model: {DEFAULT_SPACY_MODEL}")
            try:
                # Disable unnecessary components for speed
                self._nlp = spacy.load(DEFAULT_SPACY_MODEL, disable=["parser", "ner"])
                # Add common Italian stop words from spaCy
                self._italian_stopwords = self._nlp.Defaults.stop_words
                self.logger.info("spaCy model loaded.")
            except OSError:
                 self.logger.error(f"spaCy model '{DEFAULT_SPACY_MODEL}' not found. Please run: python -m spacy download {DEFAULT_SPACY_MODEL}")
                 sys.exit(1)
            except Exception as e:
                 self.logger.error(f"Failed to load spaCy model: {e}", exc_info=True)
                 sys.exit(1)


    def _load_sbert(self):
         if self._sbert_model is None:
            self.logger.info(f"Loading SentenceTransformer model: {DEFAULT_EMBEDDING_MODEL_SBERT}")
            try:
                self._sbert_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL_SBERT)
                self.logger.info("SentenceTransformer model loaded.")
            except Exception as e:
                self.logger.error(f"Failed to load SentenceTransformer model: {e}", exc_info=True)
                sys.exit(1)


    def _load_synonyms(self, synonyms_path: str) -> Dict[str, str]:
        """Loads synonym map from JSON file."""
        try:
            if Path(synonyms_path).is_file():
                with open(synonyms_path, 'r', encoding='utf-8') as f:
                    syn_map = json.load(f)
                    self.logger.info(f"Loaded {len(syn_map)} synonym rules from {synonyms_path}")
                    # Normalize keys and values in the map itself
                    return {self._normalize_basic(k): self._normalize_basic(v) for k, v in syn_map.items()}
            else:
                self.logger.warning(f"Synonyms file not found at {synonyms_path}, no synonyms will be applied.")
                return {}
        except (json.JSONDecodeError, OSError) as e:
            self.logger.error(f"Failed to load or parse synonyms file {synonyms_path}: {e}", exc_info=True)
            return {}

    def _normalize_basic(self, tag: str) -> str:
        """Basic normalization: lowercase, unidecode."""
        if not tag or not isinstance(tag, str): return ""
        # 1. Lowercase
        tag = tag.lower()
        # 2. Remove diacritics (e.g., à -> a)
        tag = unidecode.unidecode(tag)
        # 3. Keep dashes, alphanumeric; remove others (like punctuation)
        tag = re.sub(r'[^a-z0-9-]+', '', tag)
        # 4. Collapse multiple dashes
        tag = re.sub(r'-+', '-', tag)
        # 5. Remove leading/trailing dashes
        tag = tag.strip('-')
        return tag


    def normalize_single_tag(self, tag: str) -> str:
        """Applies full normalization including lemmatisation and stopword removal."""
        if self._nlp is None: self._load_nlp() # Ensure spaCy is loaded
        if not self._nlp: return "" # Failed to load

        # 1. Basic normalization
        normalized_tag = self._normalize_basic(tag)
        if not normalized_tag: return ""

        # 2. Lemmatise (handle multi-word tags separated by dashes)
        parts = normalized_tag.split('-')
        lemmatized_parts = []
        doc = self._nlp(" ".join(parts)) # Process space-separated for better tokenization

        for token in doc:
            # 3. Remove stop words and short tokens (optional: remove len <= 2)
            if token.lemma_ not in self._italian_stopwords and len(token.lemma_) > 2:
                 # Further clean lemma (remove potential leftover punctuation if any)
                lemma_clean = re.sub(r'[^a-z0-9]+', '', token.lemma_)
                if lemma_clean:
                    lemmatized_parts.append(lemma_clean)

        # Re-join with dashes if multiple parts remain
        final_tag = "-".join(lemmatized_parts)

        # Ensure no leading/trailing dashes remain after stopword removal etc.
        return final_tag.strip('-')


    def apply_synonyms(self, tags: Set[str]) -> Set[str]:
        """Applies the synonym map to a set of tags."""
        normalized_tags = set()
        for tag in tags:
             # Normalize before checking synonyms, use basic norm for lookup
            basic_norm_tag = self._normalize_basic(tag)
            canonical = self.synonyms.get(basic_norm_tag, tag) # Use original if no synonym
            normalized_tags.add(canonical)
        return normalized_tags


    def cluster_and_find_canonical(self, tags: List[str]) -> Tuple[Dict[str, str], Set[str]]:
        """
        Clusters similar tags using embeddings and HDBSCAN, determines canonical forms.
        Returns:
            - alias_to_canonical_map: Dict mapping each input tag to its determined canonical tag.
            - canonical_tags_set: Set of unique canonical tags identified.
        """
        if self._sbert_model is None: self._load_sbert() # Ensure SBERT is loaded
        if not self._sbert_model: return {tag: tag for tag in tags}, set(tags) # Failed to load

        if not tags:
            return {}, set()

        unique_tags = sorted(list(set(tags))) # Work with unique tags
        if len(unique_tags) < 2:
            # No clustering needed for 0 or 1 tag
            return {tag: tag for tag in unique_tags}, set(unique_tags)

        self.logger.debug(f"Clustering {len(unique_tags)} unique tags.")

        try:
            # 1. Get embeddings
            embeddings = self._sbert_model.encode(unique_tags, show_progress_bar=False)

            # 2. Calculate cosine similarity (convert to distance for HDBSCAN)
            # similarity_matrix = cosine_similarity(embeddings)
            # distance_matrix = 1 - similarity_matrix
            # Note: HDBSCAN works better with Euclidean distance or others.
            # Let's try default Euclidean on embeddings.

            # 3. Run HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
                metric='euclidean', # or 'cosine' if preferred with appropriate data prep
                # alpha=1.0, # Default, controls density vs distance
                # cluster_selection_epsilon=0.5 # Optional: merge clusters closer than this distance
                # allow_single_cluster=True # Allow finding just one large cluster
            )
            cluster_labels = clusterer.fit_predict(embeddings)

            num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            num_noise = np.sum(cluster_labels == -1)
            self.logger.debug(f"HDBSCAN found {num_clusters} clusters and {num_noise} noise points.")

            # 4. Determine canonical tag for each cluster
            canonical_map = {} # raw_tag -> canonical_tag
            canonical_tags = set() # Store the chosen canonicals

            for i, tag in enumerate(unique_tags):
                label = cluster_labels[i]

                if label == -1:
                    # Noise point, becomes its own canonical tag
                    canonical_map[tag] = tag
                    canonical_tags.add(tag)
                else:
                    # Belongs to a cluster
                    if label not in canonical_map:
                         # First time seeing this cluster, determine its canonical tag
                         cluster_indices = np.where(cluster_labels == label)[0]
                         cluster_tags = [unique_tags[idx] for idx in cluster_indices]

                         # Strategy for canonical: shortest tag in the cluster
                         canonical_tag = min(cluster_tags, key=len)
                         # Alternative: most frequent if frequency data available, or first alphabetically

                         self.logger.debug(f"Cluster {label}: Tags={cluster_tags} -> Canonical='{canonical_tag}'")
                         canonical_tags.add(canonical_tag)

                         # Map all tags in this cluster to the chosen canonical
                         for cluster_tag in cluster_tags:
                             canonical_map[cluster_tag] = canonical_tag
                    # else: canonical already determined for this cluster label

            return canonical_map, canonical_tags

        except Exception as e:
            self.logger.error(f"Tag clustering failed: {e}", exc_info=True)
            # Fallback: return original tags as canonical
            return {tag: tag for tag in unique_tags}, set(unique_tags)

    def normalise_and_update_aliases(self, raw_tags: List[str]) -> Set[str]:
        """
        Full normalization pipeline: basic norm -> lemmatize/stopwords -> synonyms -> clustering.
        Updates the tag_alias table in the database.
        Returns the set of canonical tag names identified for the input list.
        """
        if not raw_tags:
            return set()

        # 1. Apply normalization (lemmatize, stopwords etc.) to each raw tag
        normalized_tags = set()
        for tag in raw_tags:
            norm_tag = self.normalize_single_tag(tag)
            if norm_tag: # Only keep non-empty tags after normalization
                normalized_tags.add(norm_tag)
        self.logger.debug(f"After normalize_single_tag: {normalized_tags}")

        # 2. Apply predefined synonym rules
        tags_after_synonyms = self.apply_synonyms(normalized_tags)
        self.logger.debug(f"After apply_synonyms: {tags_after_synonyms}")


        # 3. Cluster remaining tags to find canonical forms
        alias_to_canonical_map, final_canonical_tags = self.cluster_and_find_canonical(list(tags_after_synonyms))
        self.logger.debug(f"After clustering: Canonical tags={final_canonical_tags}")
        self.logger.debug(f"Alias map from clustering: {alias_to_canonical_map}")


        # 4. Update the database with new aliases found during clustering
        # Ensure canonical tags exist in the DB first
        canonical_tag_ids = {}
        for can_tag in final_canonical_tags:
             tag_id = self.db_client.get_or_create_tag(can_tag) # This also adds self-alias
             canonical_tag_ids[can_tag] = tag_id

        # Update aliases based on clustering results
        for alias, canonical_tag_name in alias_to_canonical_map.items():
             if alias != canonical_tag_name: # Only add explicit aliases
                 canonical_tag_id = canonical_tag_ids.get(canonical_tag_name)
                 if canonical_tag_id:
                     self.db_client.add_tag_alias(alias, canonical_tag_id)
                     self.logger.debug(f"Updated DB alias: '{alias}' -> '{canonical_tag_name}' (ID: {canonical_tag_id})")


        # We return the set of canonical tag *names* determined for this document
        return final_canonical_tags



# --- Main Orchestration ---

def write_back(file_path: Path, canonical_tags: List[str], dry_run: bool) -> None:
    """Writes the canonical tags back to the note's YAML front matter."""
    logger = logging.getLogger("write_back")
    try:
        # Preserve original mtime
        original_stat = file_path.stat()

        with file_path.open('r', encoding='utf-8') as f:
            try:
                post = frontmatter.load(f)
            except ScannerError as e:
                 logger.error(f"{file_path.name} | Failed to parse YAML front matter: {e}. Skipping write-back.")
                 return
            except Exception as e: # Catch other potential loading errors
                 logger.error(f"{file_path.name} | Error loading file content: {e}. Skipping write-back.", exc_info=True)
                 return


        # Ensure 'tags' key exists and is a list
        if 'tags' not in post.metadata or not isinstance(post.metadata['tags'], list):
             post.metadata['tags'] = []

        # Sort incoming tags and ensure uniqueness
        unique_sorted_new_tags = sorted(list(set(canonical_tags)))

        # Update logic: replace existing tags completely
        # Alternative: merge (post.metadata['tags'] = sorted(list(set(post.metadata['tags']) | set(unique_sorted_new_tags))))
        post.metadata['tags'] = unique_sorted_new_tags

        logger.info(f"{file_path.name} | Updating YAML tags to: {unique_sorted_new_tags}")

        if not dry_run:
            try:
                 # Use ruamel.yaml handler for better preservation
                yaml = YAML()
                yaml.indent(mapping=2, sequence=4, offset=2)
                yaml.preserve_quotes = True

                # Dump the modified content back to the file
                updated_content = frontmatter.dumps(post, handler=frontmatter.YAMLHandler())

                with file_path.open('w', encoding='utf-8') as f:
                    f.write(updated_content)

                # Restore original modification time
                os.utime(file_path, (original_stat.st_atime, original_stat.st_mtime))
                logger.info(f"{file_path.name} | Successfully updated file and restored mtime.")

            except OSError as e:
                logger.error(f"{file_path.name} | Failed to write updated file: {e}", exc_info=True)
            except Exception as e:
                 logger.error(f"{file_path.name} | Unexpected error during write-back: {e}", exc_info=True)
        else:
            logger.info(f"{file_path.name} | [Dry Run] Skipped writing tags to file.")

    except FileNotFoundError:
        logger.error(f"{file_path.name} | File not found during write-back process.")
    except Exception as e:
        logger.error(f"{file_path.name} | Error processing file for write-back: {e}", exc_info=True)

# Find the main() function

def main():
    """Main script execution function."""
    global db_client_global

    args = parse_args()
    setup_logging(args.log_level, DEFAULT_LOG_FILE)
    logging.info("Tagger script started.")
    logging.debug(f"Arguments: {args}")

    load_env(args.env_file) # Load .env before checking keys

    # --- Validate API key for OpenAI ---
    openai_api_key = None
    if args.llm_provider == 'openai':
        openai_api_key = args.openai_api_key # Already loaded from env/arg by parse_args logic
        if not openai_api_key:
            logging.critical("OpenAI API key is required when --llm-provider=openai. Provide via --openai-api-key, .env file, or OPENAI_API_KEY environment variable.")
            sys.exit(1)
        logging.info("Using OpenAI provider.")
    elif args.llm_provider == 'ollama':
         logging.info(f"Using Ollama provider with model '{args.ollama_model}' at {args.ollama_url}")
         # API key not needed, but the library requires one - pass a dummy string
         openai_api_key = "ollama"


    # --- Initialize components ---
    cost_tracker = CostTracker(DEFAULT_PRICE_TABLE)
    # Override OpenAI prices if provided via CLI and provider is OpenAI
    if args.llm_provider == 'openai' and (args.price_prompt is not None or args.price_completion is not None):
         model_prices = cost_tracker.price_table.get(args.openai_model, {})
         if args.price_prompt is not None:
             model_prices['prompt'] = args.price_prompt
         if args.price_completion is not None:
              model_prices['completion'] = args.price_completion
         cost_tracker.price_table[args.openai_model] = model_prices
         logging.info(f"Updated price table for {args.openai_model}: {model_prices}")


    try:
        db_client = DBClient(args.db)
        db_client_global = db_client
    except Exception as e:
        logging.critical(f"Failed to initialize database. Exiting. Error: {e}", exc_info=True)
        sys.exit(1)

    if args.reset_db:
        if args.dry_run:
            logging.warning("[Dry Run] Skipping database reset.")
        else:
            db_client.reset_db()
    else:
         db_client.ensure_schema()


    # --- Initialize LLM Client (Conditionally) ---
    try:
        if args.llm_provider == 'openai':
            llm_client = OpenAI(api_key=openai_api_key)
            # Test connection (optional, but good practice)
            llm_client.models.list()
            logging.info("OpenAI client initialized and connection tested.")
            actual_model_name = args.openai_model # Set model name for OpenAI
        elif args.llm_provider == 'ollama':
            # Use OpenAI client but point to Ollama URL
            llm_client = OpenAI(
                base_url=args.ollama_url, # Already includes /v1 from parse_args
                api_key=openai_api_key, # Dummy key, required by lib but ignored by Ollama
            )
            # Optional: Add a basic connection test for Ollama if possible/needed
            # e.g., try listing models or a simple completion
            try:
                 # Simple test: List models (might require different endpoint if /v1 doesn't support)
                 # or just proceed and handle errors during first call
                 # llm_client.models.list() # This might fail depending on Ollama's OpenAI compatibility level
                 logging.info(f"Ollama client initialized pointing to {args.ollama_url}.")
            except Exception as ollama_conn_err:
                 logging.warning(f"Could not fully verify connection to Ollama during init: {ollama_conn_err}", exc_info=False)
            actual_model_name = args.ollama_model # Set model name for Ollama
        else:
             # Should not happen due to argparse choices, but belts and braces
             logging.critical(f"Unsupported LLM provider: {args.llm_provider}")
             sys.exit(1)

    except APIError as e:
         # This might catch auth errors for OpenAI, or connection errors if test fails
         logging.critical(f"Failed to initialize LLM client for {args.llm_provider}: {e}. Check API key/URL and network.", exc_info=True)
         if db_client: db_client.close()
         sys.exit(1)
    except Exception as e: # Catch other init errors
         logging.critical(f"Unexpected error initializing LLM client: {e}", exc_info=True)
         if db_client: db_client.close()
         sys.exit(1)


    # --- Initialize other components ---
    vault_walker = VaultWalker(args.vault)
    # Pass the correct provider and model name to TagExtractor
    tag_extractor = TagExtractor(
        openai_client=llm_client,
        cost_tracker=cost_tracker,
        llm_provider=args.llm_provider, # Pass provider
        model_name=actual_model_name,   # Pass the selected model name
        max_tags_per_chunk=15
    )
    tag_normaliser = TagNormaliser(db_client, args.synonyms)

    # --- Setup interrupt handler ---
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # --- Process files ---
    files_processed = 0
    files_skipped = 0
    new_tags_added_count = 0 # Count newly created canonical tags

    try:
        all_files = vault_walker.walk()
        if not all_files:
             logging.warning("No Markdown files found in the specified vault path.")

        # Wrap file iteration with tqdm for progress bar
        for file_path in tqdm(all_files, desc="Processing files", unit="file"):
            if interrupted: break # Check for interruption flag

            relative_path_str = str(file_path.relative_to(args.vault)) # Use relative path for DB
            file_name = file_path.name
            logging.debug(f"Checking file: {relative_path_str}")

            try:
                file_stat = file_path.stat()
                mtime = file_stat.st_mtime

                # --- Resume logic ---
                if args.resume:
                    status = db_client.get_doc_status(relative_path_str, mtime)
                    if status == 'done':
                        logging.debug(f"{file_name} | Already processed and mtime matches. Skipping.")
                        files_skipped += 1
                        continue
                    elif status == 'started':
                         logging.warning(f"{file_name} | Found 'started' status. Re-processing.")
                         # Continue processing
                    elif status == 'error':
                         logging.warning(f"{file_name} | Found 'error' status. Re-processing.")
                         # Continue processing
                    # Else: No record or mtime mismatch, process normally


                # --- Read file content ---
                with file_path.open('r', encoding='utf-8') as f:
                    # Read metadata first without parsing full content yet if possible
                    try:
                         post_peek = frontmatter.load(f)
                         title = post_peek.metadata.get('title') # Get title if exists
                         # Reset file pointer to read full content later if needed
                         f.seek(0)
                         full_content = f.read()
                    except Exception as e:
                         logging.error(f"{file_name} | Failed to read or parse frontmatter: {e}. Skipping file.", exc_info=True)
                         # Mark as error in DB? Requires doc_id first. Maybe skip DB ops.
                         continue


                # --- Add/Update document record ---
                doc_id = db_client.add_or_update_doc(relative_path_str, title, mtime)
                if not args.dry_run:
                    db_client.mark_status(doc_id, 'started')
                    db_client.commit() # Commit after starting work on doc

                logging.info(f"{file_name} | Starting processing (doc_id: {doc_id}).")

                # --- Tag extraction ---
                raw_tags = tag_extractor.extract(full_content, relative_path_str, args.chunk_size_chars)

                if interrupted: break # Check again after potentially long step

                # --- Tag normalisation & DB update ---
                # This step needs the DB to check/update aliases
                start_tag_count = len(db_client._fetchall("SELECT id FROM tag")) # Approximate count before
                canonical_tags_set = tag_normaliser.normalise_and_update_aliases(raw_tags)
                end_tag_count = len(db_client._fetchall("SELECT id FROM tag")) # Approximate count after
                new_tags_added_count += (end_tag_count - start_tag_count)

                logging.info(f"{file_name} | Normalised tags: {sorted(list(canonical_tags_set))}")

                if interrupted: break

                # Limit number of tags per document
                final_tags_for_doc = sorted(list(canonical_tags_set))[:args.max_tags_per_doc]

                # --- Link tags to document in DB ---
                if not args.dry_run:
                    # Clear existing links (already done in add_or_update_doc if mtime changed)
                    # self.db_client._execute("DELETE FROM doc_tag WHERE doc_id = ?", (doc_id,)) # Redundant?

                    # Link new tags
                    for tag_name in final_tags_for_doc:
                         # get_or_create_tag ensures tag exists before linking
                        tag_id = db_client.get_or_create_tag(tag_name)
                        if tag_id:
                            db_client.link_doc_tag(doc_id, tag_id)

                # --- Write back to file ---
                write_back(file_path, final_tags_for_doc, args.dry_run)

                # --- Mark as done ---
                if not args.dry_run:
                    db_client.mark_status(doc_id, 'done')
                    db_client.commit() # Commit all changes for this document
                else:
                     # Simulate commit for logging/flow if needed, but rollback
                     logging.info("[Dry Run] Skipping database commit for file.")


                files_processed += 1
                logging.info(f"{file_name} | Processing finished successfully.")


            except Exception as e:
                logging.error(f"{file_name} | Unhandled error during processing: {e}", exc_info=True)
                # Attempt to mark as error in DB if doc_id was obtained
                if 'doc_id' in locals() and doc_id and not args.dry_run:
                     try:
                         db_client.mark_status(doc_id, 'error')
                         db_client.commit()
                     except Exception as db_err:
                         logging.error(f"{file_name} | Failed to mark error status in DB: {db_err}", exc_info=True)
                # Continue to next file


    except KeyboardInterrupt:
         # Already handled by signal handler, just ensures loop breaks
         pass
    except Exception as e:
        logging.critical(f"An unexpected critical error occurred during vault processing: {e}", exc_info=True)
    finally:
        # --- Final Report ---
        logging.info("-" * 30)
        final_summary = (
             f"Finished. Processed: {files_processed} files | "
             f"Skipped: {files_skipped} files | "
             f"New canonical tags created (approx): {new_tags_added_count}"
        )
        logging.info(final_summary)
        print(final_summary)
        cost_tracker.report()

        # --- Cleanup ---
        if db_client:
            db_client.close() # Already closed by signal handler if interrupted
        if log_handler_global:
             log_handler_global.flush()
             log_handler_global.close() # Close log file handle


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: This script requires Python 3.10 or higher.", file=sys.stderr)
        sys.exit(1)
    main()