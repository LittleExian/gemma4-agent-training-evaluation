"""Expose Capybara's enabled thought channel to vLLM's token-budget processor."""


def adapt_template(source):
    if ('add_generation_prompt' not in source or '<|turn>model' not in source
            or 'CAPYBARA_BUDGET_PREFIX_V1' in source):
        raise ValueError('Unexpected or already adapted Capybara template')
    return source + "\n{# CAPYBARA_BUDGET_PREFIX_V1 #}{%- if add_generation_prompt and enable_thinking | default(false) -%}{{- '<|channel>thought\\n' -}}{%- endif -%}"
