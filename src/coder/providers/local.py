"""The measured Qwen prompt, with catalogue-constrained local transports."""
import json
from urllib.parse import urlparse
import urllib.request

from ..clinical_query import diagnosis_focus

from ..prompt import make_messages, parse_choices

class LocalReranker:
    """One sequential request; no stored responses or conversation history.

    Ollama manages its own model memory. For llama.cpp use --cache-ram 0,
    --no-cache-idle-slots and one slot, as documented in docs/RUNNING.md.
    """
    def __init__(self, backend, endpoint, model, timeout):
        parsed = urlparse(endpoint)
        if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}
                or parsed.username or parsed.password or parsed.path not in {'', '/'}
                or parsed.query or parsed.fragment):
            raise ValueError('local inference endpoint must be an HTTP loopback address')
        self.backend, self.endpoint = backend, endpoint.rstrip('/')
        self.model, self.timeout = model, timeout
        # Only loopback is allowed here; external API clients must retain proxies.
        self.client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def rerank(self, text, candidates, titles, timeout=None):
        messages = make_messages(text, candidates, titles)
        if self.backend == 'ollama':
            path = '/api/chat'
            payload = {'model': self.model, 'messages': messages, 'stream': False,
                       'format': 'json', 'keep_alive': '30m',
                       'options': {'temperature': 0, 'seed': 42, 'num_ctx': 8192, 'num_predict': 256}}
        else:
            path = '/v1/chat/completions'
            payload = {'messages': messages, 'temperature': 0, 'seed': 42,
                       'max_tokens': 256, 'stream': False, 'cache_prompt': False,
                       'response_format': {'type': 'json_object'}}
        request = urllib.request.Request(self.endpoint + path, json.dumps(payload).encode(),
                                         {'Content-Type': 'application/json'})
        with self.client.open(request, timeout=min(self.timeout, timeout or self.timeout)) as response:
            result = json.load(response)
        if self.backend == 'ollama':
            if result.get('done_reason') == 'length':
                raise ValueError('model exhausted output budget before completing JSON')
            content = result['message']['content']
        else:
            if result['choices'][0].get('finish_reason') == 'length':
                raise ValueError('model exhausted output budget before completing JSON')
            content = result['choices'][0]['message']['content']
        return parse_choices(content, candidates)
