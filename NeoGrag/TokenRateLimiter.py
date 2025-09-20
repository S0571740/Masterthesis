import time
import asyncio
import tiktoken
from collections import deque

class TokenRateLimiter:
    def __init__(self, max_tokens_per_minute):
        self.max_tokens = max_tokens_per_minute
        self.token_log = deque()  # stores (timestamp, tokens)
        self.encoding = tiktoken.get_encoding("o200k_base")  # adjust for your LLM

    def count_tokens(self, text):
        return len(self.encoding.encode(text))

    def _cleanup(self):
        """Remove token usages older than 60 seconds."""
        one_minute_ago = time.time() - 60
        while self.token_log and self.token_log[0][0] < one_minute_ago:
            self.token_log.popleft()

    async def wait_if_needed(self, token_count):
        """Check token budget, wait if needed."""
        self._cleanup()
        current_usage = sum(t for _, t in self.token_log)

        if current_usage + token_count > self.max_tokens:
            time_to_wait = 60 - (time.time() - self.token_log[0][0])
            print(f"[RateLimit] Sleeping {time_to_wait:.2f}s to respect token budget.")
            await asyncio.sleep(time_to_wait)
            self._cleanup()

        self.token_log.append((time.time(), token_count))
