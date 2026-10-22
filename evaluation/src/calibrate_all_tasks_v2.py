"""Explicitly enable snapshot-declared AnyIO tests without changing test files."""
from pathlib import Path

import calibrate_all_tasks as calibration

original_manager = calibration.isolated_manager
original_fingerprint = calibration.fingerprint_inputs


def with_declared_async_backend(manager_type, shard, test_env, transcript, target_package):
    manager = original_manager(manager_type, shard, test_env, transcript, target_package)
    execute = manager.exec

    def run(identifier, command, **kwargs):
        if target_package == 'fastapi' and 'python3 -s -m pytest ' in command:
            previous = command
            command = command.replace(' -p no:anyio', ' -p anyio -o addopts=')
            transcript.append({'phase': 'harness_configuration_override',
                               'reason': 'Run snapshot-declared @pytest.mark.anyio tests instead of disabling their plugin.',
                               'original_command': previous, 'actual_command': command})
        return execute(identifier, command, **kwargs)

    manager.exec = run
    return manager


def inputs(base):
    return {**original_fingerprint(base), 'wrapper_sha256': calibration.sha(Path(__file__)),
            'harness_overrides': {'fastapi_anyio_plugin': True,
                                  'clear_disabled_anyio_ini_addopts': True}}


calibration.isolated_manager = with_declared_async_backend
calibration.fingerprint_inputs = inputs


if __name__ == '__main__':
    calibration.main()
