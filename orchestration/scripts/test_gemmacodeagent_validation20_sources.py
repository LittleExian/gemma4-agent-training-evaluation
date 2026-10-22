import io
import unittest
from unittest.mock import patch

from gemmacodeagent_validation20_sources import RangeFile


class RangeFileTests(unittest.TestCase):
    def test_seek_read_and_readinto(self):
        source = b'0123456789'
        calls = []

        def fetch(urls, start, end, size, retries, timeout):
            calls.append((start, end))
            return source[start:end + 1]

        with patch('gemmacodeagent_validation20_sources.fetch', side_effect=fetch):
            file = RangeFile('https://example.test/file', len(source))
            self.assertEqual(file.read(3), b'012')
            self.assertEqual(file.seek(-2, io.SEEK_END), 8)
            self.assertEqual(file.read(100), b'89')
            self.assertEqual(file.read(1), b'')
            file.seek(4)
            buffer = bytearray(3)
            self.assertEqual(file.readinto(buffer), 3)
            self.assertEqual(buffer, b'456')
            self.assertEqual(calls, [(0, 2), (8, 9), (4, 6)])

    def test_negative_seek_rejected(self):
        with self.assertRaises(ValueError):
            RangeFile('https://example.test/file', 3).seek(-1)


if __name__ == '__main__':
    unittest.main()
