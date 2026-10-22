"""Unit tests; the parent runner performs Linux confinement integration tests."""
import unittest

import landlock_guard as guard


class AccessMaskTests(unittest.TestCase):
    def test_abi1_omits_later_rights(self):
        mask = guard.handled_rights(1)
        self.assertFalse(mask & guard.REFER)
        self.assertFalse(mask & guard.TRUNCATE)

    def test_abi2_handles_refer_but_not_truncate(self):
        mask = guard.handled_rights(2)
        self.assertTrue(mask & guard.REFER)
        self.assertFalse(mask & guard.TRUNCATE)

    def test_abi4_handles_all_filesystem_rights_known_to_abi4(self):
        self.assertEqual(guard.handled_rights(4), (1 << 15) - 1)

    def test_read_directory_does_not_grant_writes(self):
        self.assertEqual(guard.allowed_rights(4, False, True),
                         guard.EXECUTE | guard.READ_FILE | guard.READ_DIR)

    def test_regular_file_has_no_directory_only_rights(self):
        self.assertEqual(guard.allowed_rights(4, True, False),
                         guard.EXECUTE | guard.READ_FILE | guard.WRITE_FILE |
                         guard.TRUNCATE)

    def test_write_directory_does_not_allow_device_creation(self):
        mask = guard.allowed_rights(4, True, True)
        self.assertFalse(mask & guard.MAKE_CHAR)
        self.assertFalse(mask & guard.MAKE_BLOCK)
        self.assertTrue(mask & guard.REFER)

    def test_path_beneath_kernel_layout_is_packed(self):
        self.assertEqual(guard.ctypes.sizeof(guard.PathBeneathAttr), 12)

    def test_no_command_is_rejected(self):
        with self.assertRaises(SystemExit):
            guard.parse_args(["--read", "/usr"])


if __name__ == "__main__":
    unittest.main()
