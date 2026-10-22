"""Release only a recorded same-user process tree, preserving primary failures."""
import math
import os
import sys


def _clock_quantum():
    """Linux /proc start times are truncated to one kernel clock tick."""
    if not sys.platform.startswith('linux'):
        return 0.0
    ticks = os.sysconf('SC_CLK_TCK')
    if not isinstance(ticks, int) or ticks <= 0:
        raise RuntimeError('Cannot establish process timestamp precision')
    return 1.0 / ticks


def _matches(process, created, uid):
    return process.create_time() == created and process.uids().real == uid


def release_owned_model(record, started, *, wait_timeout=15):
    """Require exact PID, recorded ctime and UID; tolerate only kernel truncation.

    On Linux, ``record['launch_boottime']`` must be CLOCK_BOOTTIME sampled
    immediately BEFORE Popen. Linux psutil timestamps use coarse /proc btime,
    which is not the realtime epoch origin. Comparing realtime directly can
    reject a newly spawned process. Never infer this boundary after launch.
    Other platforms compare ``started`` (realtime) directly. Exact recorded
    identity checks are never relaxed.
    """
    import psutil
    pid, created, uid = record['pid'], record['create_time'], record['uid']
    if (not isinstance(pid, int) or pid <= 0 or uid != os.getuid()
            or not math.isfinite(created) or not math.isfinite(started)):
        raise ValueError('Invalid owned process identity')
    boundary, process_start = started, created
    if sys.platform.startswith('linux'):
        boundary = record.get('launch_boottime')
        if boundary is None or not math.isfinite(boundary) or boundary < 0:
            raise ValueError('Linux cleanup requires recorded pre-launch CLOCK_BOOTTIME')
        process_start = created - psutil.boot_time()
    if process_start < boundary - _clock_quantum():
        raise RuntimeError('Refusing cleanup of a process predating this launcher')
    report = {'terminated': [], 'killed': []}
    try:
        root = psutil.Process(pid)
        if not _matches(root, created, uid):
            raise RuntimeError('Recorded process PID/create_time/UID identity changed')
        targets = [(root, created)]
        for child in root.children(recursive=True):
            try:
                child_created = child.create_time()
                if child.uids().real == uid and child_created >= created:
                    targets.append((child, child_created))
            except psutil.NoSuchProcess:
                continue
        if not _matches(root, created, uid):
            raise RuntimeError('Recorded process identity changed during child snapshot')
    except psutil.NoSuchProcess:
        return report
    for process, timestamp in reversed(targets):
        try:
            if _matches(process, timestamp, uid):
                process.terminate()
                report['terminated'].append(process.pid)
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs([process for process, _ in targets], timeout=wait_timeout)
    identities = {process.pid: timestamp for process, timestamp in targets}
    for process in alive:
        try:
            if _matches(process, identities[process.pid], uid):
                process.kill()
                report['killed'].append(process.pid)
        except psutil.NoSuchProcess:
            continue
    return report


release_owned = release_owned_model


def cleanup_secondary(cleanup_callable, primary_error=None):
    """Return cleanup evidence; attach cleanup failure without masking primary."""
    try:
        return cleanup_callable()
    except Exception as error:
        if primary_error is None:
            raise
        detail = f'{type(error).__name__}: {error}'
        if hasattr(primary_error, 'add_note'):
            primary_error.add_note(f'Secondary process cleanup failed: {detail}')
        return {'cleanup_error': detail}
