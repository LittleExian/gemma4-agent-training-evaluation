import unittest

from launch_budget_full import diagnostic_counts


class DiagnosticTests(unittest.TestCase):
    def test_structured_diagnostics_are_counted_without_hashing_dicts(self):
        records = [
            {'verification_diagnostic': {'categories': ['collection_error', 'import_or_missing_dependency']}},
            {'verification_diagnostic': {'categories': []}},
            {},
        ]
        self.assertEqual(diagnostic_counts(records),
                         {'collection_error': 1, 'import_or_missing_dependency': 1, 'none': 2})


if __name__ == '__main__':
    unittest.main()
