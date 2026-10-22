import unittest

from weighted_lora_model_download import trusted_files


class ManifestTests(unittest.TestCase):
    def test_hf_lfs_oid_is_mandatory_identity(self):
        result = trusted_files([{'type': 'file', 'path': 'a.safetensors',
                                 'size': 12, 'oid': 'a' * 40,
                                 'lfs': {'oid': 'b' * 64, 'size': 12}}])
        self.assertEqual(result[0]['lfs']['sha256'], 'b' * 64)

    def test_non_lfs_uses_git_blob_hash(self):
        result = trusted_files([{'type': 'file', 'path': 'config.json',
                                 'size': 2, 'oid': 'a' * 40}])
        self.assertEqual(result[0]['blobId'], 'a' * 40)

    def test_path_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            trusted_files([{'type': 'file', 'path': '../bad',
                            'size': 2, 'oid': 'a' * 40}])

    def test_invalid_hash_is_rejected(self):
        with self.assertRaises(ValueError):
            trusted_files([{'type': 'file', 'path': 'config.json',
                            'size': 2, 'oid': 'bad'}])


if __name__ == '__main__':
    unittest.main()
