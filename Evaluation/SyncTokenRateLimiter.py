
from collections import deque
import threading
import time

import tiktoken

class SyncTokenRateLimiter:
    """
    Synchronous token bucket / sliding-window limiter.
    Call wait_for_tokens(token_count) before issuing a request.
    """

    def __init__(self, max_tokens_per_minute: int):
        self.max_tokens = max_tokens_per_minute
        self.token_log = deque()  # stores (timestamp, tokens)
        self.lock = threading.Lock()
    
    def _cleanup(self):
        one_minute_ago = time.time() - 60.0
        while self.token_log and self.token_log[0][0] < one_minute_ago:
            self.token_log.popleft()

    def current_usage(self) -> int:
        self._cleanup()
        return sum(tokens for _, tokens in self.token_log)

    def wait_for_tokens(self, token_count: int):
        """
        Blocking call: waits until token_count tokens are available within the minute budget.
        """
        if token_count <= 0:
            return
        while True:
            with self.lock:
                self._cleanup()
                usage = self.current_usage()
                if usage + token_count <= self.max_tokens:
                    # consume tokens and proceed
                    self.token_log.append((time.time(), token_count))
                    return
                # otherwise need to wait until the earliest logged usage is > 60s old
                if self.token_log:
                    oldest_ts = self.token_log[0][0]
                    wait_time = max(0.0, 60.0 - (time.time() - oldest_ts))
                else:
                    wait_time = 1.0
            # release lock and sleep before retrying
            # optional: small backoff
            time.sleep(wait_time + 0.1)

    def count_tokens(self, text: str, model: str = "text-embedding-ada-002") -> int:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    
_global_token_limiter = None
embedding_token_limiter = None
llm_token_limiter = None

def get_embedding_token_limiter(max_tokens_per_minute=1_000_000):
    global embedding_token_limiter
    if embedding_token_limiter is None:
        embedding_token_limiter = SyncTokenRateLimiter(max_tokens_per_minute)
    return embedding_token_limiter

def get_llm_token_limiter(max_tokens_per_minute=200_000):
    global llm_token_limiter
    if llm_token_limiter is None:
        llm_token_limiter = SyncTokenRateLimiter(max_tokens_per_minute)
    return llm_token_limiter