import unittest

from tools.mind import build_command


class BuildCommandTest(unittest.TestCase):
    def test_builds_generate_command(self):
        self.assertEqual(
            build_command([12, 34, 56], 20),
            b"GENERATE 20 12 34 56\n",
        )

    def test_rejects_empty_prompt(self):
        with self.assertRaisesRegex(ValueError, "zero tokens"):
            build_command([], 20)

    def test_rejects_long_prompt(self):
        with self.assertRaisesRegex(ValueError, "device maximum"):
            build_command(list(range(97)), 20)


if __name__ == "__main__":
    unittest.main()
