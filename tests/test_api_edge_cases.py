"""API edge cases use the real SDK and fake HTTP, never paid inference.

The small note/code fixtures here test protocol handling, not clinical accuracy.
Real-encoder CLI checks live in test_app_smoke and test_anthropic_wire.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import anthropic
import httpx2

from coder.__main__ import run
from coder.providers.anthropic import AnthropicReranker
from coder.reranker import Reranker
from test_anthropic_wire import reply


KEY = "test-key-never-real"
CANDIDATES = ["A", "B", "C"]
TITLES = dict(A="First test category", B="Second test category", C="Third test category")


class APIEdgeCases(unittest.TestCase):
    def api(self, response=None, status=200, **options):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx2.Response(status, json=reply() if response is None else response)

        client = anthropic.Anthropic(
            api_key=KEY, max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(respond)))
        with patch("anthropic.Anthropic", return_value=client):
            api = AnthropicReranker(KEY, "claude-haiku-5-5", 10, **options)
        self.addCleanup(api.close)
        return api, requests

    def rank(self, api, text="Protocol test", **kwargs):
        return api.rerank(text, CANDIDATES, TITLES, **kwargs)

    def test_forced_tool_requests_exactly_one_ranking(self):
        api, requests = self.api()
        self.assertEqual(self.rank(api), ["B", "A", "C"])
        body = json.loads(requests[0].content)
        self.assertTrue(body["tool_choice"].get("disable_parallel_tool_use"))
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertNotIn("temperature", body)  # Do not assume model-specific support.
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertIsInstance(body["system"], str)
        schema = body["tools"][0]["input_schema"]
        self.assertEqual(schema["required"], ["choices"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["choices"]["items"]["enum"], [1, 2, 3])

    def test_refused_paused_or_incomplete_turn_never_counts_as_success(self):
        for stop in ("refusal", "pause_turn", "end_turn", "stop_sequence", None):
            with self.subTest(stop=stop):
                api, requests = self.api(reply(stop=stop))
                with self.assertRaises(ValueError):
                    self.rank(api)
                self.assertEqual(len(requests), 1)

    def test_malformed_success_body_matrix(self):
        cases = []
        for content in (None, {}, "not a block list", [None]):
            body = reply(); body["content"] = content; cases.append(body)
        for tool_input in (None, [], "not an object", {}, {"choices": None},
                           {"choices": [1.0, 2, 3]}, {"choices": [1e90, 2, 3]},
                           {"choices": ["A", "B", "C"]}, {"choices": [-1, 2, 3]}):
            body = reply(); body["content"][0]["input"] = tool_input; cases.append(body)
        cases.extend(({}, {"type": "message"}))
        for index, body in enumerate(cases):
            with self.subTest(case=index):
                api, requests = self.api(body)
                with self.assertRaises(Exception):
                    self.rank(api)
                self.assertEqual(len(requests), 1)

    def test_explanatory_text_alongside_one_valid_tool_is_accepted(self):
        body = reply()
        body["content"].insert(0, {"type": "text", "text": "Selected candidates."})
        api, _ = self.api(body)
        self.assertEqual(self.rank(api), ["B", "A", "C"])

    def test_no_malformed_response_can_invent_a_catalogue_code(self):
        for choices in ([True, 2, 3], [1, 2, 9999], [1, 1, 2], ["1", "2", "3"]):
            with self.subTest(choices=choices):
                api, _ = self.api(reply(choices))
                router = Reranker("none", "unused", "http://127.0.0.1:11434", 1)
                router.clients = [("anthropic", api)]
                self.assertIsNone(router.rerank("Test", CANDIDATES, TITLES, 1))
                self.assertEqual(router.routes, {"retrieval": 1})

    def test_deadline_is_passed_to_the_actual_sdk_transport(self):
        api, requests = self.api()
        self.rank(api, timeout=0.25)
        self.assertLessEqual(requests[0].extensions["timeout"]["read"], 0.25)

    def test_candidate_limits_and_requests_do_not_accumulate_conversation(self):
        for size in (3, 50, 100):
            with self.subTest(size=size):
                codes = [f"TEST-{i}" for i in range(size)]
                titles = {code: code for code in codes}
                api, requests = self.api()
                api.rerank("first-unique-input", codes, titles)
                api.rerank("second-unique-input", codes, titles)
                body = json.loads(requests[1].content)
                self.assertEqual(len(body["messages"]), 1)
                self.assertNotIn("first-unique-input", requests[1].content.decode())
                numbers = body["tools"][0]["input_schema"]["properties"]["choices"]["items"]["enum"]
                self.assertEqual(numbers, list(range(1, size + 1)))

    def test_oversized_unicode_input_is_rejected_before_a_request(self):
        api, requests = self.api()
        with self.assertRaisesRegex(ValueError, "input exceeds"):
            self.rank(api, "\U0001f912" * 26_000)
        self.assertEqual(requests, [])
        self.assertEqual(api.calls, 0)
        self.assertEqual(api.reserved_usd, 0)

    def test_failed_requests_keep_their_budget_reservation(self):
        api, requests = self.api(status=503)
        for _ in range(2):
            with self.assertRaises(anthropic.APIStatusError):
                self.rank(api)
        self.assertEqual(len(requests), 2)
        self.assertGreater(api.reserved_usd, 0)
        api.budget_usd = api.reserved_usd
        with self.assertRaisesRegex(ValueError, "spend reservation"):
            self.rank(api)
        self.assertEqual(len(requests), 2)

    def test_explicit_api_key_and_official_url_override_other_auth_settings(self):
        with patch.dict(os.environ, {"ANTHROPIC_AUTH_TOKEN": "wrong-token",
                                     "ANTHROPIC_BASE_URL": "https://wrong.invalid"}):
            client = anthropic.Anthropic(api_key=KEY, base_url="https://api.anthropic.com")
            try:
                self.assertEqual(str(client.base_url), "https://api.anthropic.com")
                self.assertEqual(client.auth_headers["Authorization"], "Bearer " + KEY)
            finally:
                client.close()

    def test_routing_modes_do_not_contact_unselected_providers(self):
        for backend, environment, expected in (
            ("auto", {"ANTHROPIC_API_KEY": KEY, "QWEN_BACKEND": "ollama"}, ["anthropic", "qwen"]),
            ("auto", {"ANTHROPIC_API_KEY": KEY}, ["anthropic"]),
            ("auto", {"QWEN_BACKEND": "ollama"}, ["qwen"]),
            ("auto", {}, []),
            ("ollama", {"ANTHROPIC_API_KEY": KEY}, ["qwen"]),
            ("none", {"ANTHROPIC_API_KEY": KEY, "QWEN_BACKEND": "ollama"}, []),
        ):
            with self.subTest(backend=backend, environment=list(environment)):
                with patch.dict(os.environ, environment, clear=True), \
                     patch("coder.providers.anthropic.AnthropicReranker") as api:
                    router = Reranker(backend, "qwen", "http://127.0.0.1:11434", 1)
                self.assertEqual([name for name, _ in router.clients], expected)
                self.assertEqual(api.call_count, int("anthropic" in expected))

    def test_invalid_budget_or_prices_disable_only_the_api(self):
        cases = [{name: value} for name in (
            "CODER_API_BUDGET_USD", "ANTHROPIC_INPUT_USD_PER_MILLION",
            "ANTHROPIC_OUTPUT_USD_PER_MILLION")
            for value in ("not-a-number", "nan", "inf", "-1", "0")]
        cases += [{"CODER_API_BUDGET_USD": "6.01"}, {"ANTHROPIC_MODEL": "custom-model"}]
        for settings in cases:
            with self.subTest(settings=settings), patch.dict(
                os.environ, {"ANTHROPIC_API_KEY": KEY, "QWEN_BACKEND": "ollama", **settings}, clear=True):
                router = Reranker("auto", "qwen", "http://127.0.0.1:11434", 1)
                self.assertEqual([name for name, _ in router.clients], ["qwen"])
                self.assertIn("anthropic_setup", router.errors)

    def test_custom_model_and_explicit_prices_reach_the_sdk(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": KEY,
                "ANTHROPIC_MODEL": "custom-model", "ANTHROPIC_INPUT_USD_PER_MILLION": "1",
                "ANTHROPIC_OUTPUT_USD_PER_MILLION": "5"}, clear=True), \
             patch("coder.providers.anthropic.AnthropicReranker") as api:
            router = Reranker("auto", "qwen", "http://127.0.0.1:11434", 10)
        api.assert_called_once_with(KEY, "custom-model", 10, 5.5, 1, 5)
        self.assertEqual([name for name, _ in router.clients], ["anthropic"])

    def test_an_intermediate_success_resets_consecutive_failures(self):
        router = Reranker("none", "unused", "http://127.0.0.1:11434", 1)
        api = Mock()
        api.rerank.side_effect = [TimeoutError(), TimeoutError(), ["B", "A", "C"],
                                 TimeoutError(), TimeoutError(), ["B", "A", "C"]]
        router.clients = [("anthropic", api)]
        results = [router.rerank("Test", CANDIDATES, TITLES, 1) for _ in range(6)]
        self.assertEqual(results[2], ["B", "A", "C"])
        self.assertEqual(results[5], ["B", "A", "C"])
        self.assertEqual(api.rerank.call_count, 6)


class APIOutputPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.jsonl"
        self.output = self.root / "output.jsonl"
        self.catalogue = self.root / "catalogue.json"
        self.catalogue.write_text(json.dumps([{"code": code, "title": title}
                                             for code, title in TITLES.items()]))
        self.notes = [{"id": name, "text": "Protocol test", "gold": ["DO-NOT-SEND"]}
                      for name in ("one", "二", "tres")]
        self.source.write_text("".join(json.dumps(row) + "\n" for row in self.notes))

    def simulate(self, responses, environment=None):
        requests = []
        queue = iter(responses)

        def respond(request):
            requests.append(request)
            self.assertNotIn("DO-NOT-SEND", request.content.decode())
            status, body = next(queue)
            return httpx2.Response(status, json=body)

        constructor = anthropic.Anthropic

        def factory(**kwargs):
            return constructor(**kwargs, http_client=httpx2.Client(
                transport=httpx2.MockTransport(respond)))

        hybrid = Mock(language_counts={"en": 3})
        hybrid.retrieve_many.return_value = ([CANDIDATES[:]] * 3, [CANDIDATES[:]] * 3)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": KEY, **(environment or {})}, clear=True), \
             patch("anthropic.Anthropic", side_effect=factory), \
             patch("coder.__main__.load_hybrid", return_value=hybrid):
            summary = run(self.source, self.output, self.catalogue)
        rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual([row["id"] for row in rows], [row["id"] for row in self.notes])
        self.assertTrue(all(len(row["codes"]) == len(set(row["codes"])) == 3 and
                            set(row["codes"]) <= set(CANDIDATES) for row in rows))
        self.assertNotIn(KEY, json.dumps(summary))
        return summary, requests

    def test_extended_account_http_failure_matrix_keeps_all_outputs(self):
        for status in (400, 401, 402, 403, 404, 408, 409, 413, 422, 429, 500, 502, 503, 529):
            body = {"type": "error", "error": {"type": "api_error", "message": "simulated"}}
            with self.subTest(status=status):
                summary, requests = self.simulate([(status, body)] * 3)
                self.assertEqual(summary["routes"], {"retrieval": 3})
                self.assertEqual(len(requests), 3)

    def test_bad_json_choice_then_valid_rankings_continue(self):
        summary, requests = self.simulate([(200, reply([1, 1, 2])), (200, reply()), (200, reply())])
        self.assertEqual(summary["routes"], {"retrieval": 1, "anthropic": 2})
        self.assertEqual(len(requests), 3)

    def test_tiny_budget_never_contacts_provider_and_keeps_outputs(self):
        summary, requests = self.simulate([], {"CODER_API_BUDGET_USD": "0.000001"})
        self.assertEqual(requests, [])
        self.assertEqual(summary["api_calls"], 0)
        self.assertEqual(summary["routes"], {"retrieval": 3})

    def test_gold_and_extra_fields_do_not_reach_api(self):
        summary, requests = self.simulate([(200, reply())] * 3)
        self.assertEqual(summary["routes"], {"anthropic": 3})
        self.assertEqual(len(requests), 3)

    def test_generated_predictions_can_be_evaluated_by_the_supplied_scorer(self):
        self.simulate([(200, reply())] * 3)  # [2,1,3] must map to B,A,C.
        gold = self.root / "gold.jsonl"
        gold.write_text("".join(json.dumps({"id": row["id"], "gold": ["B"]}) + "\n"
                                for row in self.notes))
        scorer = Path(__file__).resolve().parents[1] / "score.py"
        result = subprocess.run([sys.executable, str(scorer), "--gold", str(gold),
                                 "--pred", str(self.output), "--mms", str(self.catalogue)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("missing predictions 0  invalid codes 0  duplicate ids 0", result.stdout)
        self.assertIn("top1 1.000  top3 1.000", result.stdout)
        self.assertIn("SCORE 75.0 / 75", result.stdout)

    def test_invalid_input_is_rejected_before_api_setup_or_output_truncation(self):
        original = b"keep-existing-output\n"
        self.output.write_bytes(original)
        cases = ("{bad", "[]", '{"id":null,"text":"test"}',
                 '{"id":"one","text":null}', '{"id":"one"}',
                 '{"id":"one","text":"x"}\n{"id":"one","text":"y"}')
        for content in cases:
            with self.subTest(content=content):
                self.source.write_text(content)
                with patch("coder.providers.anthropic.AnthropicReranker") as api:
                    with self.assertRaises(ValueError):
                        run(self.source, self.output, self.catalogue)
                    api.assert_not_called()
                self.assertEqual(self.output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
