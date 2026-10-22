"""Use calibrated scoring, stopping new dispatch on infrastructure/control failure."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import gemmacodeagent_full_run as runner


def fail_fast_rolling(tasks, slots, work, complete):
    pending = iter(tasks)
    failed = False
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        active = {}
        for endpoint in slots:
            task = next(pending, None)
            if task is not None:
                active[pool.submit(work, task, endpoint)] = endpoint
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            endpoints = []
            for future in done:
                endpoints.append(active.pop(future))
                record = future.result()
                complete(record)
                failed = failed or record.get('status') in ('infra_error', 'control_failed')
            if not failed:
                for endpoint in endpoints:
                    task = next(pending, None)
                    if task is not None:
                        active[pool.submit(work, task, endpoint)] = endpoint
    if failed:
        raise RuntimeError('Infrastructure/control failure: stopped dispatch; preserved in-flight outcomes')


if __name__ == '__main__':
    runner.rolling = fail_fast_rolling
    runner.main()
