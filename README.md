# Obsidian Tagger

This Python script automatically scans an Obsidian vault, extracts relevant tags from Markdown notes using KeyBERT and OpenAI's GPT models, normalizes them, and writes them back into the YAML front matter of each note. It uses an SQLite database to track progress and manage tags, allowing for graceful resuming after interruptions.

## Features

* Scans Obsidian vaults (`.md` files), skipping the `.obsidian` configuration directory.
* Follows symbolic links.
* Preserves existing YAML front matter, adding or updating a `tags` list.
* Cleans note content by removing code blocks, HTML, and YAML before analysis.
* Chunks large notes to fit context window limits while trying to respect paragraph boundaries.
* Uses **KeyBERT** for initial keyword extraction.
* Uses **OpenAI GPT models** (e.g., `gpt-4o`) to generate relevant tags based on content and KeyBERT suggestions (in Italian).
* **Normalizes** tags: lowercase, removes diacritics, lemmatizes (using `spaCy` Italian model), removes stopwords/punctuation.
* Applies a customizable **synonym map** (JSON).
* Uses **Sentence-BERT embeddings** and **HDBSCAN clustering** to identify and collapse near-duplicate tags into a single canonical form.
* Stores document info, tags, aliases, and processing status in an **SQLite database**.
* **Resumable:** Skips files that have already been processed successfully (based on path and modification time).
* Updates the `tags:` list in the note's YAML front matter with the final canonical tags.
* Preserves the original file modification time.
* Detailed **logging** to console and a rotating file (`tagger.log`).
* Tracks **OpenAI API costs**.
* Configurable via command-line arguments.
* Handles `Ctrl+C` (SIGINT) for graceful shutdown.
* Includes basic unit tests.

## Installation

1.  **Clone the repository or download the files:**
    * `tagger.py`
    * `synonyms.json` (customize as needed)
    * `README.md` (this file)
    * `tests/` folder (optional, for running tests)

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS
    source venv/bin/activate
    # On Windows (cmd/PowerShell)
    # .\venv\Scripts\activate
    ```

3.  **Upgrade pip and install required packages:**
    ```bash
    python -m pip install --upgrade pip
    pip install spacy==3.* frontmatter ruamel.yaml keybert sentence-transformers openai unidecode hdbscan scikit-learn tqdm python-dotenv pytest numpy
    ```
    *Note: Ensure `numpy` is installed, as it's a dependency for `hdbscan` and embedding processing.*

4.  **Download the Italian spaCy model:**
    ```bash
    python -m spacy download it_core_news_lg
    ```

5.  **Set up OpenAI API Key:**
    Create a file named `.env` in the same directory as `tagger.py` with your API key:
    ```dotenv
    # .env
    OPENAI_API_KEY="sk-..."
    ```
    Alternatively, you can set the `OPENAI_API_KEY` environment variable or provide it via the `--openai-api-key` command-line argument.

## Usage

Run the script from your terminal, providing the path to your Obsidian vault.

```bash
python tagger.py --vault /path/to/your/obsidian/vault [OPTIONS]