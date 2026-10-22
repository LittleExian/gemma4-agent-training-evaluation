import hashlib
import contextlib
import io
import http.server
import importlib
import json
import os
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path

m = importlib.import_module('gemmacodeagent_mm_download')


def spec(data=b'abcdefghij', name='model.safetensors'):
    return {'rfilename': name, 'size': len(data), 'lfs': {'sha256': hashlib.sha256(data).hexdigest()}}


class Handler(http.server.BaseHTTPRequestHandler):
    data = b'abcdefghij'
    bad_range = False
    requests = []

    def do_GET(self):
        span = self.headers.get('Range')
        self.requests.append(span)
        if span:
            start, end = map(int, span[6:].split('-'))
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(self.data) + int(self.bad_range)}')
            self.end_headers()
            self.wfile.write(self.data[start:end + 1])
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(self.data)

    def log_message(self, *args):
        pass


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        Handler.bad_range = False
        Handler.requests = []
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/file'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def test_git_blob_hash_and_sha256(self):
        data = b'abcdefghij'
        path = self.root / 'config.json'
        path.write_bytes(data)
        info = {'size': len(data), 'blobId': m.git_blob(data)}
        self.assertTrue(m.verified(path, info))
        self.assertTrue(m.verified(path, spec()))
        self.assertFalse(m.verified(path, spec(b'bad')))

    def test_safe_files_and_invalid_paths(self):
        source = {'subfolder': 'FP8/', 'files': [spec(name='FP8/model.safetensors'), spec(name='FP8/evil.py'), spec(name='FP8/config.json')]}
        self.assertEqual(len(m.selected_files(source)), 2)
        for name in ['../x', '/x', 'FP8/../x', 'x\\y']:
            with self.assertRaises(ValueError):
                m.safe_name(name)

    def test_symlink_and_hardlink_rejected(self):
        real = self.root / 'real'
        real.write_bytes(b'abcdefghij')
        link = self.root / 'link'
        link.symlink_to(real)
        with self.assertRaises(ValueError):
            m.verified(link, spec())
        hard = self.root / 'hard'
        os.link(real, hard)
        with self.assertRaises(ValueError):
            m.verified(hard, spec())

    def test_small_download_and_skip(self):
        dest = self.root / 'config.json'
        info = {'rfilename': dest.name, 'size': 10, 'blobId': m.git_blob(Handler.data)}
        m.download_file(dest, info, [self.url], chunk_size=16, retries=1)
        self.assertEqual(dest.read_bytes(), Handler.data)
        m.download_file(dest, info, [self.url], chunk_size=16, retries=1)
        self.assertEqual(len(Handler.requests), 1)

    def test_range_resume_and_hash(self):
        dest = self.root / 'model.safetensors'
        part = dest.with_name(dest.name + '.partial')
        part.write_bytes(b'abcd' + b'\0' * 6)
        journal = dest.with_name(dest.name + '.chunks.json')
        journal.write_text(json.dumps({'identity': m.identity(spec()), 'chunk_size': 4, 'done': [0]}))
        m.download_file(dest, spec(), [self.url], chunk_size=4, workers=2, retries=1)
        self.assertEqual(dest.read_bytes(), Handler.data)
        self.assertNotIn('bytes=0-3', Handler.requests)

    def test_existing_invalid_never_overwritten(self):
        dest = self.root / 'model.safetensors'
        dest.write_bytes(b'wrong')
        with self.assertRaises(ValueError):
            m.download_file(dest, spec(), [self.url], retries=1)
        self.assertEqual(dest.read_bytes(), b'wrong')

    def test_bad_range_and_wrong_hash_fail_closed(self):
        Handler.bad_range = True
        with self.assertRaises(RuntimeError):
            m.download_file(self.root / 'm', spec(), [self.url], chunk_size=4, retries=1)
        Handler.bad_range = False
        with self.assertRaises(ValueError):
            m.download_file(self.root / 'n', spec(b'XXXXXXXXXX'), [self.url], chunk_size=4, retries=1)
        self.assertFalse((self.root / 'n').exists())

    def test_sources_immutable_and_mirror_urls(self):
        source = {'repo': 'Qwen/X', 'revision': 'a' * 40, 'subfolder': ''}
        spec_model = {'modelscope_repo': 'Qwen/M'}
        urls = m.file_urls(source, spec_model, spec(), ['modelscope', 'hf-mirror'])
        self.assertIn('modelscope.cn/models/Qwen/M/resolve/master/', urls[0])
        self.assertIn('/resolve/' + 'a' * 40, urls[1])
        with self.assertRaises(ValueError):
            m.file_urls({**source, 'revision': 'main'}, spec_model, spec(), ['hf'])

    def test_directory_symlink_rejected(self):
        (self.root / 'target').mkdir()
        (self.root / 'alias').symlink_to(self.root / 'target')
        with self.assertRaises(ValueError):
            m.safe_directory(self.root / 'alias' / 'child')

    def test_full_plan_and_download_cli_integration(self):
        entry = {'model': {'directory': 'tiny'}, 'source': {'repo': 'A/B', 'revision': 'a' * 40},
                 'files': [spec()], 'bytes': 10}
        with mock.patch.object(m, 'load_pinned', return_value={'tiny_arm': entry}), contextlib.redirect_stdout(io.StringIO()) as out:
            m.main(['plan', '--experiment-dir', str(self.root)])
            self.assertEqual(json.loads(out.getvalue())['tiny_arm']['bytes'], 10)
            with mock.patch.object(m, 'file_urls', return_value=[self.url]):
                m.main(['download', '--experiment-dir', str(self.root), '--cache-dir', str(self.root / 'cache')])
        target = self.root / 'cache' / 'tiny'
        self.assertEqual((target / 'model.safetensors').read_bytes(), Handler.data)
        self.assertEqual(json.loads((target / 'download-status.json').read_text())['status'], 'complete')
        self.assertEqual(json.loads((target / 'DOWNLOAD_COMPLETE.json').read_text())['upstream_commit'], m.COMMIT)

    def test_manifest_source_revisions_are_pinned(self):
        model = {'a': {'directory': 'tiny', 'repo': 'A/B', 'subfolder': ''}}
        raw_model = json.dumps(model).encode()
        source = {'repo': 'A/B', 'subfolder': '', 'revision': 'a' * 40, 'files': [spec()]}
        raw_source = json.dumps(source).encode()
        (self.root / 'models.json').write_bytes(raw_model)
        (self.root / 'sources').mkdir()
        sourcepath = self.root / 'sources' / 'tiny_source.json'
        sourcepath.write_bytes(raw_source)
        with mock.patch.object(m, 'MODELS_BLOB', m.git_blob(raw_model)), mock.patch.object(m, 'SOURCE_BLOBS', {'tiny': m.git_blob(raw_source)}):
            self.assertEqual(m.load_pinned(self.root)['a']['bytes'], 10)
            sourcepath.write_text('{}')
            with self.assertRaises(ValueError):
                m.load_pinned(self.root)
        with self.assertRaises(ValueError):
            m.load_pinned(self.root)

    def test_resume_manifest_mismatch_and_chunk_indices(self):
        dest = self.root / 'model.safetensors'
        (self.root / 'model.safetensors.partial').write_bytes(b'0' * 10)
        journal = self.root / 'model.safetensors.chunks.json'
        for state in [{'identity': {}, 'chunk_size': 4, 'done': []},
                      {'identity': m.identity(spec()), 'chunk_size': 4, 'done': [-1]}]:
            journal.write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                m.download_file(dest, spec(), [self.url], chunk_size=4, retries=1)

    def test_atomic_json_rejects_link_and_bad_identity(self):
        path = self.root / 'state.json'
        path.symlink_to(self.root / 'absent')
        with self.assertRaises(ValueError):
            m.atomic_json(path, {})
        for info in [{'size': -1}, {'size': 10, 'blobId': 'oops'}]:
            with self.assertRaises(ValueError):
                m.identity(info)
        with self.assertRaises(ValueError):
            m.selected_files({'subfolder': '', 'files': [spec(name='evil.py')]})

    def test_transport_fallback_and_no_credential_exception(self):
        bad = self.url.replace('/file', '/error')
        with mock.patch.object(m.urllib.request, 'urlopen', side_effect=OSError('secret-token')):
            with self.assertRaisesRegex(RuntimeError, '^All bounded transport retries failed$'):
                m.fetch([bad], None, 0, 10, retries=1, timeout=1)
        data = m.fetch([self.url], None, 0, 10, retries=1, timeout=1)
        self.assertEqual(data, Handler.data)


if __name__ == '__main__':
    unittest.main()
