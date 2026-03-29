"""Unix Domain Socket IPC for waydeeper daemon management."""

import json
import logging
import os
import socket
import tempfile
import threading
from pathlib import Path
from typing import Callable, Any

logger = logging.getLogger(__name__)

RUNTIME_DIR = Path(tempfile.gettempdir()) / f"waydeeper-{os.getuid()}"


class DaemonSocket:
    """Socket server for daemon to accept commands."""

    def __init__(self, monitor: str, command_handler: Callable[[str, dict], Any]):
        self.monitor = monitor
        self.command_handler = command_handler
        self.socket_path = RUNTIME_DIR / f"{monitor}.sock"
        self.server = None
        self.running = False
        self.thread = None

    def start(self):
        """Start listening on socket."""
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

        # Remove old socket if exists
        if self.socket_path.exists():
            try:
                # Test if another daemon is using it
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect(str(self.socket_path))
                sock.close()
                raise RuntimeError(
                    f"Another daemon already running for monitor {self.monitor}"
                )
            except (socket.error, ConnectionRefusedError):
                # Socket is stale, remove it
                self.socket_path.unlink()

        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.socket_path))
        self.server.listen(5)
        self.running = True

        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()

        logger.info(f"IPC socket listening at {self.socket_path}")

    def stop(self):
        """Stop listening and cleanup socket."""
        self.running = False
        if self.server:
            self.server.close()
        if self.socket_path.exists():
            self.socket_path.unlink()
        logger.info("IPC socket stopped")

    def _accept_loop(self):
        """Accept and handle incoming connections."""
        while self.running:
            try:
                self.server.settimeout(1.0)
                conn, addr = self.server.accept()
                self._handle_connection(conn)
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_connection(self, conn: socket.socket):
        """Handle a single client connection."""
        try:
            conn.settimeout(5.0)
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            if not data:
                return

            # Parse request
            request = json.loads(data.decode().strip())
            command = request.get("command", "")
            params = request.get("params", {})

            # Handle command
            result = self.command_handler(command, params)

            # Send response
            response = {"success": True, "result": result}
            conn.sendall(json.dumps(response).encode() + b"\n")

        except json.JSONDecodeError as e:
            response = {"success": False, "error": f"Invalid JSON: {e}"}
            conn.sendall(json.dumps(response).encode() + b"\n")
        except Exception as e:
            response = {"success": False, "error": str(e)}
            conn.sendall(json.dumps(response).encode() + b"\n")
        finally:
            conn.close()


class DaemonClient:
    """Client for communicating with daemon via socket."""

    def __init__(self, monitor: str):
        self.monitor = monitor
        self.socket_path = RUNTIME_DIR / f"{monitor}.sock"

    def is_running(self) -> bool:
        """Check if daemon is responsive."""
        if not self.socket_path.exists():
            return False
        try:
            response = self.send_command("PING", {}, timeout=2)
            return response.get("success", False)
        except:
            return False

    def send_command(
        self, command: str, params: dict = None, timeout: float = 5
    ) -> dict:
        """Send a command to the daemon and return response."""
        if not self.socket_path.exists():
            raise ConnectionRefusedError(
                f"Daemon not running for monitor {self.monitor}"
            )

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            sock.connect(str(self.socket_path))

            request = {"command": command, "params": params or {}}
            sock.sendall(json.dumps(request).encode() + b"\n")

            # Receive response
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            response = json.loads(data.decode().strip())
            return response

        finally:
            sock.close()

    def get_status(self) -> dict:
        """Get daemon status."""
        return self.send_command("STATUS")

    def stop_daemon(self) -> bool:
        """Request daemon to stop."""
        try:
            response = self.send_command("STOP")
            return response.get("success", False)
        except:
            return False


def list_running_daemons() -> list[str]:
    """List all monitors with responsive daemons."""
    if not RUNTIME_DIR.exists():
        return []

    running = []
    for sock_path in RUNTIME_DIR.glob("*.sock"):
        monitor = sock_path.stem
        client = DaemonClient(monitor)
        if client.is_running():
            running.append(monitor)
        else:
            # Cleanup stale socket
            try:
                sock_path.unlink()
            except:
                pass

    return running


def get_daemon_info(monitor: str) -> dict | None:
    """Get info about a running daemon."""
    client = DaemonClient(monitor)
    if not client.is_running():
        return None

    try:
        response = client.get_status()
        if response.get("success"):
            return response.get("result")
    except Exception as e:
        logger.warning(f"Failed to get info for {monitor}: {e}")

    return None


def stop_daemon(monitor: str, timeout: float = 5) -> bool:
    """Stop a daemon gracefully via socket."""
    client = DaemonClient(monitor)
    if not client.is_running():
        return False

    try:
        # Send stop command - if we get a success response, daemon will stop
        if client.stop_daemon():
            # Wait a moment for daemon to start shutting down
            import time

            time.sleep(0.5)
            # Verify daemon is no longer responsive
            for _ in range(int((timeout - 0.5) * 10)):
                if not client.is_running():
                    return True
                time.sleep(0.1)

        return False
    except Exception as e:
        logger.warning(f"Failed to stop daemon {monitor}: {e}")
        return False


def stop_all_daemons(timeout: float = 5) -> dict[str, bool]:
    """Stop all running daemons."""
    results = {}
    for monitor in list_running_daemons():
        results[monitor] = stop_daemon(monitor, timeout)
    return results
