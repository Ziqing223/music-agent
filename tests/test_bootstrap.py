import unittest

import music_agent


class BootstrapTest(unittest.TestCase):
    def test_package_is_importable_and_versioned(self) -> None:
        self.assertEqual(music_agent.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
