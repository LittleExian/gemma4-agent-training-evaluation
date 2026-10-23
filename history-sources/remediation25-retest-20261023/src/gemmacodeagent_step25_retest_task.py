"""Explicit seeded request sampling over the audited self-contained task adapter."""
import gemmacodeagent_mm_task as legacy
import gemmacodeagent_morning25_task as audited


def adapt_source(source, context):
    adapted = audited.adapt_source(source, context)
    before = "extra_body={'thinking_token_budget': 6000, 'top_k': 64, 'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': False}}"
    after = "seed=42, extra_body={'thinking_token_budget': 6000, 'top_k': -1, 'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': False}}"
    if adapted.count(before) != 2:
        raise ValueError('Expected exactly two audited request routes')
    return adapted.replace(before, after)


def main():
    legacy.adapt_source = adapt_source
    legacy.main()


if __name__ == '__main__':
    main()
