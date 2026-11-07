"""Anthropic SDK client with a conservative per-run spend reservation."""
import json

from ..prompt import make_messages, parse_choices


class AnthropicReranker:
    def __init__(self, api_key, model, timeout, budget_usd=5.5,
                 input_usd_per_million=0.5, output_usd_per_million=2.5):
        from anthropic import Anthropic

        # Explicit official URL; SDK retains HTTPS_PROXY and certificate trust.
        self.client = Anthropic(api_key=api_key, base_url='https://api.anthropic.com',
                                max_retries=0, timeout=timeout)
        self.model, self.timeout = model, timeout
        self.budget_usd, self.reserved_usd = budget_usd, 0.0
        self.input_rate, self.output_rate = input_usd_per_million, output_usd_per_million
        self.calls = 0

    def rerank(self, text, candidates, titles, timeout=None):
        messages = make_messages(text, candidates, titles)
        # UTF-8 bytes + protocol/tool overhead overestimates normal text tokens.
        # Reserve the whole amount even on failures, with no hidden SDK retries.
        input_bound = len(json.dumps(messages, ensure_ascii=False).encode()) + 2048
        if input_bound > 100_000:
            raise ValueError('API input exceeds conservative request budget')
        reserve = (input_bound * self.input_rate + 256 * self.output_rate) / 1_000_000
        if self.reserved_usd + reserve > self.budget_usd:
            raise ValueError('API spend reservation reached; use fallback')
        self.reserved_usd += reserve
        self.calls += 1
        result = self.client.messages.create(
            model=self.model, max_tokens=256, thinking={'type': 'disabled'},
            system=messages[0]['content'], messages=messages[1:],
            timeout=min(self.timeout, timeout or self.timeout),
            tools=[{'name': 'rank_icd_codes', 'description': 'Return the three best candidate numbers.',
                    'input_schema': {'type': 'object', 'properties': {
                        'choices': {'type': 'array', 'items': {'type': 'integer', 'enum': list(range(1, len(candidates) + 1))},
                                    'minItems': 3, 'maxItems': 3}},
                        'required': ['choices'], 'additionalProperties': False}}],
            tool_choice={'type': 'tool', 'name': 'rank_icd_codes',
                         'disable_parallel_tool_use': True},
        )
        if result.stop_reason == 'max_tokens':
            raise ValueError('API output exhausted its token budget')
        if result.stop_reason != 'tool_use':
            raise ValueError(f'API did not complete the ranking tool: {result.stop_reason}')
        blocks = [block for block in result.content if block.type == 'tool_use' and block.name == 'rank_icd_codes']
        if len(blocks) != 1:
            raise ValueError('API did not return the requested ranking tool')
        return parse_choices(json.dumps(blocks[0].input), candidates)

    def close(self):
        self.client.close()
