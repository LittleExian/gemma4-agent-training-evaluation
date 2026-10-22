from pathlib import Path
import unittest

from gemmacodeagent_capybara_template import adapt_template


class TemplateTests(unittest.TestCase):
    def test_append_only_and_guards_both_flags(self):
        source = "prefix{% if add_generation_prompt %}{{ '<|turn>model\\n' }}{% endif %}"
        adapted = adapt_template(source)
        self.assertTrue(adapted.startswith(source))
        self.assertIn('add_generation_prompt and enable_thinking', adapted)
        self.assertIn('<|channel>thought', adapted)

    def test_existing_disabled_thinking_branch_is_preserved(self):
        source = "add_generation_prompt <|turn>model {% if not enable_thinking %}<|channel>thought\\n<channel|>{% endif %}"
        result = adapt_template(source)
        self.assertTrue(result.startswith(source))
        with self.assertRaises(ValueError): adapt_template(result)

    def test_unknown_or_already_modified_template_refused(self):
        for source in ['unknown', 'add_generation_prompt <|channel>thought']:
            with self.assertRaises(ValueError): adapt_template(source)


if __name__ == '__main__': unittest.main()
