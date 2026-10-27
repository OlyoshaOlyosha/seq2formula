"""Universal launcher for seq2formula.

Works on Windows, macOS, and Linux. Finds the first available port
(starting from 8002), opens the browser, and starts the FastAPI server.
"""

import socket
import threading
import time
import webbrowser

import uvicorn


def find_free_port(start_port: int = 8002) -> int:
    """Return the first available port, starting from start_port."""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # connect_ex returns 0 if the port is in use
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free ports found in range 8002-8101")


def open_browser(url: str) -> None:
    time.sleep(1.5)  # Give uvicorn a moment to bind the port
    webbrowser.open(url)


if __name__ == "__main__":
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    
    print(f"Starting seq2formula on {url}")
    
    # Open the browser in a background thread
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    
    # Start the server
    uvicorn.run("app:app", host="127.0.0.1", port=port, log_level="info")