"""In-container rewrite trace (#42): runs prepare_prompt() with the exact
German party request that failed live at 13:33 (invented animals), printing
every tool call + result and the final prompt to stdout.

Run INSIDE the bot container (needs /app/.env + the resident chat model —
it is a separate process, its output does NOT go to dev.log):

    MSYS_NO_PATHCONV=1 docker exec humbelchat-bot \
        python /app/scripts/image_rewrite_trace.py

Optional: pass a different request as argv[1]. Expect for the party request:
5 named characters with their Visual Anchors, ZERO invented animals.
"""
import asyncio
import json
import sys

from dotenv import load_dotenv

load_dotenv("/app/.env")  # container paths — do not run this on the host


async def main() -> None:
    import logging
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    import bot_core.agent_loop as al
    from bot_core import images

    orig = al.execute_tool

    async def traced(name, args):
        result = await orig(name, args)
        print(f"\n=== TOOL {name} {json.dumps(args, ensure_ascii=False)}",
              flush=True)
        head = result[:400].replace("\n", " ")
        print(f"--- RESULT ({len(result)} chars): {head}…", flush=True)
        return result

    al.execute_tool = traced  # run_agent_loop resolves the module global

    request = (sys.argv[1] if len(sys.argv) > 1 else
               "Die komplette Party an Spieler:innen sitzt am Lagerfeuer "
               "und brät sich Marshmallows")
    print(f"=== REQUEST: {request}", flush=True)
    prompt = await images.prepare_prompt(request)
    print("\n=== FINAL PROMPT ===", flush=True)
    print(prompt, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
