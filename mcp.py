from langchain_core.tools import Tool
import requests

DEX_MCP_URL = "http://127.0.0.1:5000"  # adjust if port is different

def navigate_to_url(url: str) -> str:
    resp = requests.post(f"{DEX_MCP_URL}/navigate", json={"url": url})
    return f"Navigation result: {resp.status_code}"

def get_active_tab_screenshot(_: str) -> str:
    resp = requests.get(f"{DEX_MCP_URL}/screenshot")
    return f"Screenshot taken. Status: {resp.status_code}"

navigate_tool = Tool(
    name="DexNavigateTool",
    func=navigate_to_url,
    description="Navigate the browser to a specific URL."
)

screenshot_tool = Tool(
    name="DexScreenshotTool",
    func=get_active_tab_screenshot,
    description="Take a screenshot of the current browser tab."
)
