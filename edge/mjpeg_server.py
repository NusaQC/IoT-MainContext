import json
import time
import threading
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional
from edge.camera import CameraManager


class MJPEGHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler serving live MJPEG stream, instant snapshots,
    and system health checks for NusaQC Edge Node.
    """

    camera: Optional[CameraManager] = None
    gpio_controller = None

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed_path = self.path.split("?")[0]

        if parsed_path in ["/stream", "/video_feed"]:
            self._serve_mjpeg_stream()
        elif parsed_path == "/snapshot":
            self._serve_snapshot()
        elif parsed_path == "/health":
            self._serve_health()
        elif parsed_path == "/status":
            self._serve_status()
        elif parsed_path == "/actuate":
            self._serve_actuate()
        elif parsed_path == "/":
            self._serve_index()
        else:
            self.send_response(404)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}\n')

    def _serve_mjpeg_stream(self):
        if not self.camera:
            self.send_response(503)
            self.end_headers()
            return

        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            for frame_chunk in self.camera.generate_mjpeg(target_fps=15):
                self.wfile.write(frame_chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_snapshot(self):
        if not self.camera:
            self.send_response(503)
            self.end_headers()
            return

        jpeg = self.camera.get_snapshot_jpeg(quality=90)
        if jpeg is None:
            self.send_response(500)
            self.end_headers()
            return

        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(jpeg)

    def _serve_health(self):
        camera_active = self.camera.is_camera_active() if self.camera else False
        is_mock = self.camera.is_mock() if self.camera else False

        payload = {
            "status": "ok",
            "camera_active": camera_active,
            "mock_camera": is_mock,
            "edge_timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_status(self):
        gpio_status = self.gpio_controller.get_status() if self.gpio_controller else {}
        camera_active = self.camera.is_camera_active() if self.camera else False

        payload = {
            "node": "NusaQC-RPi4-Edge",
            "camera_active": camera_active,
            "hardware": gpio_status,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_actuate(self):
        if not self.gpio_controller:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error": "GPIO controller unavailable"}\n')
            return

        query_components = parse_qs(urlparse(self.path).query)
        signal = query_components.get("signal", ["GREEN"])[0].upper()

        new_status = self.gpio_controller.trigger_signal(signal)
        payload = {
            "action": "actuate",
            "requested_signal": signal,
            "hardware": new_status,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_index(self):
        html = """<!DOCTYPE html>
<html>
<head>
    <title>NusaQC Edge Camera Feed</title>
    <style>
        body { font-family: sans-serif; background: #121417; color: #eee; text-align: center; margin: 0; padding: 20px; }
        h1 { color: #00e5ff; }
        .stream-box { border: 2px solid #00e5ff; display: inline-block; border-radius: 8px; overflow: hidden; }
        img { display: block; max-width: 90vw; height: auto; }
        .links { margin-top: 15px; }
        a { color: #00e5ff; text-decoration: none; margin: 0 10px; }
    </style>
</head>
<body>
    <h1>NusaQC Edge Node Camera Stream</h1>
    <div class="stream-box">
        <img src="/stream" alt="Live Stream" />
    </div>
    <div class="links">
        <a href="/snapshot" target="_blank">[Snapshot JPEG]</a>
        <a href="/health" target="_blank">[Health Check]</a>
        <a href="/status" target="_blank">[GPIO Status]</a>
    </div>
</body>
</html>"""
        body = html.encode("utf-8")
        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Suppress routine MJPEG frame log spam
        if args and str(args[0]).startswith("GET /stream"):
            return
        super().log_message(format, *args)


class MJPEGServer:
    """Threaded HTTP server wrapper for the MJPEG stream service."""

    def __init__(self, camera: CameraManager, gpio_controller=None, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.camera = camera
        self.gpio_controller = gpio_controller
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Starts the HTTP server on a daemon background thread."""
        handler_class = MJPEGHandler
        handler_class.camera = self.camera
        handler_class.gpio_controller = self.gpio_controller

        self._server = ThreadingHTTPServer((self.host, self.port), handler_class)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="MJPEGServerThread")
        self._thread.start()
        print(f"🌐 [MJPEG SERVER] Live stream listening on http://{self.host}:{self.port}/stream")
        print(f"🩺 [MJPEG SERVER] Health endpoint: http://{self.host}:{self.port}/health")

    def stop(self):
        """Shuts down the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print("🌐 [MJPEG SERVER] Stream server stopped.")


# Module helper functions
_GLOBAL_SERVER: Optional[MJPEGServer] = None


def start_mjpeg_server(camera: CameraManager, gpio_controller=None, host: str = "0.0.0.0", port: int = 8080) -> MJPEGServer:
    global _GLOBAL_SERVER
    if _GLOBAL_SERVER is not None:
        _GLOBAL_SERVER.stop()
    _GLOBAL_SERVER = MJPEGServer(camera=camera, gpio_controller=gpio_controller, host=host, port=port)
    _GLOBAL_SERVER.start()
    return _GLOBAL_SERVER


def stop_mjpeg_server():
    global _GLOBAL_SERVER
    if _GLOBAL_SERVER is not None:
        _GLOBAL_SERVER.stop()
        _GLOBAL_SERVER = None
