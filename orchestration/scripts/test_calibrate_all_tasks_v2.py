import unittest
from unittest.mock import patch

import calibrate_all_tasks_v2 as wrapper


class AsyncBackendTests(unittest.TestCase):
    def check(self, package):
        commands = []

        class Manager:
            def exec(self, identifier, command, **kwargs):
                commands.append(command)
                return 'result'

        transcript = []
        with patch.object(wrapper, 'original_manager', return_value=Manager()):
            manager = wrapper.with_declared_async_backend(None, None, None, transcript, package)
            self.assertEqual(manager.exec('id', 'python3 -s -m pytest tests/a.py -p no:anyio -q'), 'result')
        return commands, transcript

    def test_declared_async_tests_are_enabled_and_override_is_recorded(self):
        commands, transcript = self.check('fastapi')
        self.assertIn('-p anyio -o addopts=', commands[0])
        self.assertNotIn('-p no:anyio', commands[0])
        self.assertEqual(transcript[0]['phase'], 'harness_configuration_override')

    def test_other_repositories_keep_their_original_runner_configuration(self):
        commands, transcript = self.check('rich')
        self.assertIn('-p no:anyio', commands[0])
        self.assertEqual(transcript, [])


if __name__ == '__main__':
    unittest.main()
