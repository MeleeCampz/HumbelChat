import asyncio
import pathlib
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Mock environment variables before importing anything that uses them
os.environ["DISCORD_BOT_TOKEN"] = "fake-token"
os.environ["INFER_URL"] = "http://localhost:11434/v1"
os.environ["INFER_API_KEY"] = "fake-key"
os.environ["BOT_PREFIX"] = "!ai"
os.environ["KB_PATH"] = "kb_test"

import main
import bot_core
import config.settings
import config.characters
from kb.reader import read_kb_files

async def test_logic():
    print("--- Starting Logic Test ---")
    
    # 1. Test Character Loading
    print("Testing character loading...")
    char_path = pathlib.Path("characters.json")
    if not char_path.exists():
        content = '{"characters": {"system": {"display": "System", "model": "gpt-4"}}, "default": "system"}'
        char_path.write_text(content)
    
    config.characters.load_characters(char_path)
    system_char = config.characters.get_character("system")
    assert system_char is not None, "System character should be loaded"
    print(f"  Successfully loaded: {system_char.display}")

    # 2. Test KB Reading (if directory exists)
    print("Testing KB reading...")
    kb_test_path = pathlib.Path("kb_test")
    kb_test_path.mkdir(exist_ok=True)
    test_file = kb_test_path / "test.txt"
    test_file.write_text("This is a test knowledge base entry.")
    
    docs = read_kb_files(kb_test_path)
    assert len(docs) > 0, "Should have found att least one file"
    print(f"  Successfully read KB: {docs[0][0]}")

    # 3. Test bot_core.ask_ai (with mocked OpenAI)
    print("Testing bot_core.ask_api (with mocked OpenAI)...")
    
    # Use patch as a synchronous context manager
    with patch("bot_core.AsyncOpenAI") as MockOpenAI:
        mock_client = MagicMock()
        # Setup the chain: client.chat.completions.create(...)
        mock_completion = AsyncMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Hello from AI!"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        MockOpenAI.return_value = mock_client

        reply, extra = await bot_core.ask_ai(
            user_message="Hello!",
            model_slug="test-model",
            guild_id=123,
            channel_id=456,
            username="Tester"
        )
        
        assert "Hello from AI!" in reply
        assert extra["model_used"] == "test-model"
        print(f"  AI Reply: {reply}")

    # Cleanup
    if char_path.exists(): char_path.unlink()
    if kb_test_path.exists():
        for f in kb_test_path.glob("*"): f.unlink()
        kb_test_path.rmdir()

    print("--- All tests passed! ---")

if __name__ == "__main__":
    asyncio.run(test_logic())
