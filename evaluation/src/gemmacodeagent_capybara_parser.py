"""Gemma4 parser for the explicitly prefilled Capybara thinking template.

Loaded through vLLM's external reasoning-parser-plugin option. The template
must append <|channel>thought\n when enable_thinking is true. No installed
vLLM files, model weights, sampling settings, or test scoring are changed.
"""
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.gemma4_reasoning_parser import Gemma4ReasoningParser


@ReasoningParserManager.register_module('capybara_budget')
class CapybaraBudgetReasoningParser(Gemma4ReasoningParser):
    """Account for the thinking start token residing in the prompt."""

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        template_kwargs = kwargs.get('chat_template_kwargs') or {}
        self._prefilled_thinking = template_kwargs.get('enable_thinking') is True

    def extract_reasoning(self, model_output, request):
        template_kwargs = request.chat_template_kwargs or {}
        enabled = template_kwargs.get('enable_thinking', self._prefilled_thinking)
        if enabled and self.start_token not in model_output and self.end_token not in model_output:
            reasoning = model_output.removeprefix('thought\n')
            return reasoning, None
        return super().extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self, previous_text, current_text, delta_text,
        previous_token_ids, current_token_ids, delta_token_ids,
    ):
        if (self._prefilled_thinking and
                self.start_token_id not in current_token_ids):
            previous_token_ids = [self.start_token_id, *previous_token_ids]
            current_token_ids = [self.start_token_id, *current_token_ids]
        result = super().extract_reasoning_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, delta_token_ids,
        )
        # Native Gemma4 prefix buffering suppresses an empty reasoning delta,
        # including its content when the boundary starts a combined chunk.
        if (self._prefilled_thinking and result is None and
                self.end_token_id in delta_token_ids):
            _, boundary, content = delta_text.partition(self.end_token)
            if boundary and content:
                return DeltaMessage(content=content)
        return result
