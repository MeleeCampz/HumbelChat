import tempfile
import unittest
import pathlib
from kb.reader import read_kb_files, get_relevant_chunks

class TestKBReader(unittest.TestCase):
    def test_subfolder_reading(self):
        # Hermetic: build a temp KB root with a nested subfolder instead of
        # relying on a fixture directory inside the repo's kb/ package.
        with tempfile.TemporaryDirectory() as tmp:
            sub = pathlib.Path(tmp) / "nested" / "deeper"
            sub.mkdir(parents=True)
            (sub / "note.txt").write_text("Subfolder content", encoding="utf-8")

            files = read_kb_files(tmp)

            # Check if any file from the subfolder was found
            found = False
            for display_name, content in files:
                if "Subfolder content" in content:
                    found = True
                    break

            self.assertTrue(found, f"Failed to find file in subfolder. Found files: {files}")

    def test_window_clamped_to_section(self):
        """Regression: line windows used to bleed across markdown headers,
        so a query anchored on the last table of one class section pulled in
        the *next* class's spell tables (causing the model to invent spell
        slots for non-spellcasting classes).  Windows must stop at headers.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            doc = pathlib.Path(tmp) / "classes.md"
            # The matching line sits near the END of the Rogue section, so an
            # unclamped ±80 window would spill into the Sorcerer section below.
            rogue_lines = ["## Rogue"] + [
                f"rogue filler line {i}" for i in range(100)
            ] + ["rogue table row with sneakattack"]
            sorc_lines = ["## Sorcerer"] + [
                f"sorcerer spell slot row {i}" for i in range(100)
            ]
            doc.write_text("\n".join(rogue_lines + sorc_lines), encoding="utf-8")

            chunks = get_relevant_chunks(
                tmp, ["classes"], query="sneakattack", window_lines=80,
            )
            self.assertTrue(chunks, "no chunks returned")
            text = "\n".join(c for _, c in chunks)
            # The anchor content is preserved...
            self.assertIn("rogue table row with sneakattack", text)
            # ...but the window must not bleed into the next section.
            self.assertNotIn("## Sorcerer", text)
            self.assertNotIn("sorcerer spell slot row", text)

if __name__ == "__main__":
    unittest.main()
