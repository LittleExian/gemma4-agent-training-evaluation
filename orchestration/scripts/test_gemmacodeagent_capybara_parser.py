"""CPU contract tests against the installed vLLM parser implementation."""
from types import SimpleNamespace

import pytest

from gemmacodeagent_capybara_parser import CapybaraBudgetReasoningParser
from vllm.reasoning.gemma4_reasoning_parser import Gemma4ReasoningParser


class Tokenizer:
    def get_vocab(self):
        return {'<|channel>': 100, '<channel|>': 101, '<|turn>': 105,
                '<|tool_call>': 106, '<|tool_response>': 107}


def parser(enabled=True):
    return CapybaraBudgetReasoningParser(
        Tokenizer(), chat_template_kwargs={'enable_thinking': enabled})


def request(enabled=True):
    return SimpleNamespace(chat_template_kwargs={'enable_thinking': enabled})


@pytest.mark.parametrize('output,expected', [
    ('Plan', ('Plan', None)), ('', ('', None)),
    ('thought\nPlan', ('Plan', None)),
    ('Plan<channel|>Answer', ('Plan', 'Answer')),
    ('<|channel>thought\nPlan<channel|>Answer', ('Plan', 'Answer')),
    ('<|channel>thought\nPlan', ('Plan', None)),
    ('<|channel>thought\nOuter<|channel>Inner<channel|>Answer',
     ('Outer<|channel>Inner', 'Answer')),
    ('Plan<channel|><|tool_call>call:run_command{}<tool_call|>',
     ('Plan', '<|tool_call>call:run_command{}<tool_call|>')),
])
def test_nonstream_prefilled(output, expected):
    assert parser().extract_reasoning(output, request()) == expected


def test_disabled_preserves_native_content():
    assert parser(False).extract_reasoning('Answer', request(False)) == (None, 'Answer')


def test_stream_prefilled_reasoning():
    delta = parser().extract_reasoning_streaming('', 'Plan', 'Plan', [], [500], [500])
    assert delta.reasoning == 'Plan' and delta.content is None


def test_stream_boundary_then_content():
    p = parser()
    delta = p.extract_reasoning_streaming(
        'Plan', 'Plan<channel|>Answer', '<channel|>Answer',
        [500], [500, 101, 501], [101, 501])
    assert not delta.reasoning and delta.content == 'Answer'
    delta = p.extract_reasoning_streaming(
        'Plan<channel|>Answer', 'Plan<channel|>Answer!', '!',
        [500, 101, 501], [500, 101, 501, 502], [502])
    assert delta.content == '!' and delta.reasoning is None


def test_stream_explicit_start_not_duplicated():
    delta = parser().extract_reasoning_streaming(
        '', '<|channel>thought\nPlan', '<|channel>thought\nPlan',
        [], [100, 510, 511], [100, 510, 511])
    native = Gemma4ReasoningParser(Tokenizer()).extract_reasoning_streaming(
        '', '<|channel>thought\nPlan', '<|channel>thought\nPlan',
        [], [100, 510, 511], [100, 510, 511])
    assert delta == native


def test_stream_disabled_is_content():
    delta = parser(False).extract_reasoning_streaming('', 'Answer', 'Answer', [], [500], [500])
    assert delta.content == 'Answer' and delta.reasoning is None


def test_inputs_not_mutated():
    previous, current, delta = [], [500], [500]
    parser().extract_reasoning_streaming('', 'Plan', 'Plan', previous, current, delta)
    assert previous == [] and current == [500] and delta == [500]


def test_stream_disabled_boundary_preserves_native():
    args = ('', '<channel|>Answer', '<channel|>Answer', [], [101, 500], [101, 500])
    assert parser(False).extract_reasoning_streaming(*args) == (
        Gemma4ReasoningParser(Tokenizer()).extract_reasoning_streaming(*args))
