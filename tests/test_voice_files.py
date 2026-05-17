import unittest
from pathlib import Path

import voice_files


class TestVoiceFiles(unittest.TestCase):
    def test_get_output_subdir_returns_relative_path(self):
        self.assertEqual(
            voice_files._get_output_subdir(voice_files.NO_VOICE_OPTION),
            Path("no_voice"),
        )
        self.assertEqual(
            voice_files._get_output_subdir("subdir/sample.wav"),
            Path("subdir/sample"),
        )
        self.assertEqual(
            voice_files._get_output_subdir("voice.ogg"),
            Path("voice"),
        )


if __name__ == "__main__":
    unittest.main()
