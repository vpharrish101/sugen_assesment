"""API first, local Qwen second. The caller owns the retrieval fallback."""
from collections import Counter
import os
import math
import time
import sys

from .providers.local import LocalReranker


class Reranker:
    def __init__(self, backend, model, endpoint, timeout):
        self.clients = []
        self.failures = Counter()
        self.routes = Counter()
        self.errors = {}
        self.api = None
        self._api_key = os.getenv('ANTHROPIC_API_KEY', '')
        if backend in {'auto', 'anthropic'} and os.getenv('ANTHROPIC_API_KEY'):
            try:
                from .providers.anthropic import AnthropicReranker
                api_model = os.getenv('ANTHROPIC_MODEL', 'claude-haiku-5-5')
                input_rate = float(os.getenv('ANTHROPIC_INPUT_USD_PER_MILLION', '0.5'))
                output_rate = float(os.getenv('ANTHROPIC_OUTPUT_USD_PER_MILLION', '2.5'))
                budget = float(os.getenv('CODER_API_BUDGET_USD', '5.5'))
                if not all(math.isfinite(x) for x in (budget, input_rate, output_rate)) or not 0 < budget <= 6 or not input_rate > 0 or not output_rate > 0:
                    raise ValueError('API budget must be 0..6 USD and token rates positive')
                if api_model != 'claude-haiku-5-5' and not (
                    os.getenv('ANTHROPIC_INPUT_USD_PER_MILLION') and os.getenv('ANTHROPIC_OUTPUT_USD_PER_MILLION')
                ):
                    raise ValueError('set token prices explicitly when changing the API model')
                self.api = AnthropicReranker(os.environ['ANTHROPIC_API_KEY'], api_model, timeout,
                                             budget, input_rate, output_rate)
                self.clients.append(('anthropic', self.api))
            except Exception as exc:
                self.record_error('anthropic_setup', exc)
        # Automatic mode uses Qwen only if the user explicitly configures it.
        # No local model is downloaded or launched during grading.
        local_backend = backend if backend in {'ollama', 'llama'} else os.getenv('QWEN_BACKEND', '')
        if backend != 'none' and local_backend:
            try:
                if local_backend not in {'ollama', 'llama'}:
                    raise ValueError('QWEN_BACKEND must be ollama or llama')
                local_model = os.getenv('QWEN_MODEL', model)
                local_endpoint = os.getenv('QWEN_ENDPOINT', endpoint)
                client = LocalReranker(local_backend, local_endpoint, local_model, timeout)
                self.clients.append(('qwen', client))
            except Exception as exc:
                self.record_error('qwen_setup', exc)

    def record_error(self, stage, exc):
        detail = f'{type(exc).__name__}: {exc}'
        if self._api_key:
            detail = detail.replace(self._api_key, '[REDACTED]')
        self.errors[stage] = detail
        print(f'coder: {stage}: {detail}', file=sys.stderr)

    def rerank(self, text, candidates, titles, timeout):
        started = time.monotonic()
        for name, client in self.clients:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0 or self.failures[name] >= 3:
                continue
            try:
                codes = client.rerank(text, candidates, titles, timeout=remaining)
                self.failures[name] = 0
                self.routes[name] += 1
                return codes
            except Exception as exc:
                # SDK errors and local transport/validation failures share the
                # same fallback boundary. Interrupts still propagate normally.
                self.failures[name] += 1
                self.record_error(name, exc)
        self.routes['retrieval'] += 1
        return None

    def close(self):
        if self.api:
            try:
                self.api.close()
            except Exception as exc:
                self.record_error('anthropic_close', exc)
