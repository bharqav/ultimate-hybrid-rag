import os
import unittest

from config.settings import Settings


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        settings = Settings()
        self.assertEqual(settings.api_port, 8000)

    def test_env_override(self):
        os.environ["RAG_API_PORT"] = "9000"
        settings = Settings.from_env()
        self.assertEqual(settings.api_port, 9000)
        del os.environ["RAG_API_PORT"]


if __name__ == "__main__":
    unittest.main()
