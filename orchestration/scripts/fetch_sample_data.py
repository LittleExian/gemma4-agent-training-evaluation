"""Download the original notebook's two samples and shared test resources."""
import concurrent.futures
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent
data = ROOT / "data"
manifest = json.loads((data / "manifest.json").read_text())
tasks = [json.loads(line) for line in (data / "tasks.jsonl").read_text().splitlines()][:2]
selected = [entry for entry in manifest if (
    entry["name"].startswith(("sample_submission/", "wheels/", "scripts/", "docker/"))
    or entry["name"] in {f"snapshots/{task['instance_id']}.tgz" for task in tasks}
    or (entry["name"].startswith(("graphs/", "embeddings/"))
        and any(task["base_commit"] in entry["name"] for task in tasks))
)]


def fetch(entry):
    destination = data / entry["name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == entry["size"]:
        return entry["name"]
    api = KaggleApi()
    api.authenticate()
    api.competition_download_file("gemma-4-developer-agent", entry["name"],
                                  path=str(destination.parent), quiet=True)
    assert destination.stat().st_size == entry["size"], entry["name"]
    return entry["name"]


print(f"Downloading {len(selected)} files, {sum(x['size'] for x in selected)} bytes", flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    futures = [pool.submit(fetch, entry) for entry in selected]
    for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
        print(f"[{index}/{len(selected)}] {future.result()}", flush=True)
(data / "sample_manifest.json").write_text(json.dumps(selected, indent=2))
print("SAMPLE DATA COMPLETE", flush=True)
