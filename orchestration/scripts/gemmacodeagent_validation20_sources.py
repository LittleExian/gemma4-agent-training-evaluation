"""Read only selected rows from immutable official HF Parquet sources."""
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request

from gemmacodeagent_mm_download import atomic_json, fetch, safe_directory, safe_name


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.downloaded_bytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        position = origin[whence] + offset
        if position < 0:
            raise ValueError('Negative seek')
        self.position = position
        return position

    def read(self, size=-1):
        length = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if length <= 0:
            return b''
        start, end = self.position, self.position + length - 1
        data = fetch([self.url], start, end, self.size, retries=3, timeout=45)
        self.position += len(data)
        self.downloaded_bytes += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def official_tree(repo, revision):
    url = f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&limit=100'
    with urllib.request.urlopen(url, timeout=45) as response:
        data = json.load(response)
        if response.headers.get('Link') and 'rel="next"' in response.headers['Link']:
            raise ValueError('Paginated manifest needs explicit handling')
    return data


def extract_shard(repo, revision, info, selected, output):
    import pyarrow.parquet as pq
    filename = info['path']
    safe_name(filename)
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(filename)}'
    stream = RangeFile(url, info['size'])
    parquet = pq.ParquetFile(stream)
    r2e = repo.startswith('R2E-Gym/')
    column = 'commit_hash' if r2e else 'instance_id'
    found = []
    for group in range(parquet.num_row_groups):
        identifiers = parquet.read_row_group(group, columns=[column]).column(0).to_pylist()
        desired = {item['base_commit'].rstrip('^') if r2e else item['task_id']: item
                   for item in selected}
        positions = [(index, desired[value]) for index, value in enumerate(identifiers)
                     if value in desired]
        if not positions:
            continue
        table = parquet.read_row_group(group)
        for position, item in positions:
            row = table.slice(position, 1).to_pylist()[0]
            path = output / (item['task_id'] + '.source.json')
            if path.exists():
                raise ValueError('Duplicate selected task: ' + item['task_id'])
            record = {'selection': item, 'source': row,
                      'provenance': {'dataset': repo, 'revision': revision,
                                     'parquet_file': filename, 'parquet_oid': info['oid'],
                                     'parquet_lfs': info.get('lfs'), 'row_group': group,
                                     'row_within_group': position,
                                     'transport': 'official HF immutable revision HTTP ranges'}}
            atomic_json(path, record)
            print(json.dumps({'event': 'row_ready', 'task': item['task_id'],
                              'output': str(path)}), flush=True)
            found.append(item['task_id'])
    return {'parquet_file': filename, 'found': found,
            'transport_bytes': stream.downloaded_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    selected = json.loads(args.selection.read_text())
    output = safe_directory(args.output)
    sources = sorted({(item['source_dataset'], item['source_revision']) for item in selected})
    work = []
    for repo, revision in sources:
        group = [item for item in selected if item['source_dataset'] == repo]
        tree = official_tree(repo, revision)
        atomic_json(output / (repo.replace('/', '__') + '.tree.json'), tree)
        for info in tree:
            if info['type'] == 'file' and info['path'].endswith('.parquet'):
                # The filtered split duplicates old SWE-rebench test rows.
                if repo == 'nebius/SWE-rebench' and '/filtered-' in info['path']:
                    continue
                work.append((repo, revision, info, group, output))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda item: extract_shard(*item), work))
    found = [task for item in results for task in item['found']]
    if sorted(found) != sorted(item['task_id'] for item in selected):
        raise ValueError('Missing or duplicated selected task IDs')
    atomic_json(output / 'EXTRACTION_COMPLETE.json',
                {'tasks': found, 'shards': results,
                 'selection_sha256': hashlib.sha256(args.selection.read_bytes()).hexdigest()})


if __name__ == '__main__':
    main()
