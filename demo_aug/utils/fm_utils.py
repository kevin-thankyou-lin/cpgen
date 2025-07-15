import hashlib
import sqlite3


class LLMCache:
    def __init__(self, db_path: str = "cache/llm_cache.db"):
        self.db_path = db_path
        self.conn = None  # Will be initialized in __enter__

    def __enter__(self):
        """Open the database connection when entering the context."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._create_table()
        return self  # Allows `with LLMCache() as cache:` syntax

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensure the connection is properly closed when exiting the context."""
        if self.conn:
            self.conn.commit()  # Ensure all writes are saved
            self.conn.close()

    def _create_table(self):
        """Creates a table for caching LLM responses if it doesn't exist."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                prompt_hash TEXT PRIMARY KEY,
                response TEXT
            )
            """
        )
        self.conn.commit()

    def hash_prompt(self, prompt: str) -> str:
        """Generate a SHA-256 hash for the prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str):
        """Retrieve cached LLM response if available."""
        prompt_hash = self.hash_prompt(prompt)
        self.cursor.execute(
            "SELECT response FROM cache WHERE prompt_hash = ?", (prompt_hash,)
        )
        result = self.cursor.fetchone()
        return result[0] if result else None

    def set(self, prompt: str, response: str):
        """Store LLM response in cache."""
        prompt_hash = self.hash_prompt(prompt)
        self.cursor.execute(
            "INSERT OR REPLACE INTO cache (prompt_hash, response) VALUES (?, ?)",
            (prompt_hash, response),
        )
