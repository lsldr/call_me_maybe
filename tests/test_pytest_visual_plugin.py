"""Exercise the opt-in pytest visualization in an isolated subprocess."""

import json
from pathlib import Path

import pytest


def test_visual_plugin_streams_tokens_and_preserves_explicit_observers(
    pytester: pytest.Pytester,
) -> None:
    """The child run gets live token output and a valid completed trace."""
    plugin_path = Path(__file__).with_name("conftest.py").resolve()
    project_root = Path(__file__).parents[1].resolve()
    pytester.makeconftest(
        "import sys\n"
        f"sys.path.insert(0, {str(project_root)!r})\n"
        f"exec(compile(open({str(plugin_path)!r}).read(), "
        f"{str(plugin_path)!r}, 'exec'))\n"
    )
    pytester.makepyfile(
        test_child="""
        import numpy as np
        from src.decoding.decoder import Decoder
        from src.vocab.byte_table import build_byte_to_unicode
        from src.vocab.vocab_table import VocabTable

        def make_decoder():
            mapping = build_byte_to_unicode()
            raw = [bytes([value]) for value in range(256)]
            vocab = {''.join(mapping[b] for b in token): index
                     for index, token in enumerate(raw)}
            return Decoder(vocab=VocabTable(
                vocab=vocab, tokenizer={}, n_logits=len(raw),
            ))

        def test_decode_is_observed():
            decoder = make_decoder()
            callbacks = []
            decoder.trace_observer = lambda event: callbacks.append('global')
            quote = ord('"')
            scores = np.zeros(decoder.vocab.n_logits, dtype=np.float32)
            scores[quote] = 1.0
            explicit = lambda event: callbacks.append('explicit')
            decoder.decode(
                [1], lambda _: scores, 0,
                lambda state, raw: state + len(raw),
                lambda state: {quote}, lambda state: state == 1,
                namespace='visual-test', trace=explicit,
            )
            assert callbacks == ['explicit']
            decoder.decode(
                [1], lambda _: scores, 0,
                lambda state, raw: state + len(raw),
                lambda state: {quote}, lambda state: state == 1,
                namespace='visual-test-global',
            )
            assert callbacks == ['explicit', 'global']
        """
    )

    result = pytester.runpytest_subprocess("--visual", "-s", "-q")
    result.assert_outcomes(passed=1)
    assert "kind=token" in result.stderr.str()

    traces = list((pytester.path / "data" / "traces").glob("*.jsonl"))
    assert len(traces) == 1
    records = [json.loads(line) for line in traces[0].read_text().splitlines()]
    kinds = [record["kind"] for record in records]
    assert kinds[0] == "run_start"
    assert kinds[-1] == "run_end"
    assert kinds.count("attempt_start") == 2
    assert kinds.count("attempt_end") == 2
    assert kinds.count("token") == 2
