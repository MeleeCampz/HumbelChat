import unittest
import pathlib
from kb.reader import read_kb_files

class TestKBReader(unittest.TestCase):
    def test_subfolder_reading(self):
        # The path is the root of our KB (from settings.py or default)
        # We'll use 'kb' as it contains our test subfolder
        kb_path = "kb"
        files = read_kb_files(kb_path)
        
        # Check if any file from the subfolder was found
        found = False
        for display_name, content in files:
            if "Subfolder content" in content:
                found = True
                break
        
        self.assertTrue(found, f"Failed to find file in subfolder. Found files: {files}")

if __name__ == "__main__":
    unittest.main()
