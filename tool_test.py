import asyncio
import sys
import os
from pathlib import Path

# Set env vars
os.environ["QUILLAN_BASE"] = r"C:\02_QUILLAN"
os.environ["QUILLAN_MCP_BASE"] = r"C:\02_QUILLAN\09 - Projects\mcp"

mcp_dir = Path(r"C:\02_QUILLAN\09 - Projects\mcp")
sys.path.insert(0, str(mcp_dir))
sys.path.insert(0, r"C:\02_QUILLAN\09 - Projects\projects\oni")
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
sys.path.insert(0, r"C:\02_QUILLAN")


async def test_council():
    print("Testing quillan_council...")
    spec = __import__("importlib.util").util.spec_from_file_location(
        "quillan_council", r"C:\02_QUILLAN\09 - Projects\mcp\quillan_council\server.py"
    )
    mod = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Test council_full_deliberation
    result = await mod.mcp.call_tool(
        "council_full_deliberation", {"topic": "test topic", "persist": False}
    )
    print(f"  council_full_deliberation: {type(result)}")
    if hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
        for item in result:
            if hasattr(item, "text"):
                print(f"    {item.text[:200]}...")
                break
    return True


async def test_code():
    print("Testing quillan_code...")
    spec = __import__("importlib.util").util.spec_from_file_location(
        "quillan_code", r"C:\02_QUILLAN\09 - Projects\mcp\quillan_code\server.py"
    )
    mod = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Test codeweaver_ast_inspect
    test_code = "def hello():\n    print('world')\n"
    result = await mod.mcp.call_tool(
        "codeweaver_ast_inspect", {"code_or_file_path": test_code}
    )
    print(f"  codeweaver_ast_inspect: {type(result)}")
    if hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
        for item in result:
            if hasattr(item, "text"):
                print(f"    {item.text[:200]}...")
                break
    return True


async def test_media():
    print("Testing quillan_media...")
    spec = __import__("importlib.util").util.spec_from_file_location(
        "quillan_media", r"C:\02_QUILLAN\09 - Projects\mcp\quillan_media\server.py"
    )
    mod = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Test media_image_brief
    result = await mod.mcp.call_tool("media_image_brief", {"objective": "test image"})
    print(f"  media_image_brief: {type(result)}")
    if hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
        for item in result:
            if hasattr(item, "text"):
                print(f"    {item.text[:200]}...")
                break
    return True


async def test_orchestrator():
    print("Testing quillan_orchestrator...")
    spec = __import__("importlib.util").util.spec_from_file_location(
        "quillan_orchestrator",
        r"C:\02_QUILLAN\09 - Projects\mcp\quillan_orchestrator\server.py",
    )
    mod = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Test orchestrator_health
    result = await mod.mcp.call_tool("orchestrator_health", {})
    print(f"  orchestrator_health: {type(result)}")
    if hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
        for item in result:
            if hasattr(item, "text"):
                print(f"    {item.text[:300]}...")
                break
    return True


async def main():
    print("=" * 50)
    print("Testing tool calls on fixed servers")
    print("=" * 50)

    try:
        await test_council()
        print()
    except Exception as e:
        print(f"  ERROR: {e}\n")

    try:
        await test_code()
        print()
    except Exception as e:
        print(f"  ERROR: {e}\n")

    try:
        await test_media()
        print()
    except Exception as e:
        print(f"  ERROR: {e}\n")

    try:
        await test_orchestrator()
        print()
    except Exception as e:
        print(f"  ERROR: {e}\n")

    print("=" * 50)
    print("Tool call tests complete")
    print("=" * 50)


asyncio.run(main())
