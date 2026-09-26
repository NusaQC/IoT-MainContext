import time
import queue
import threading
import requests
from typing import Optional, Dict, Any


class TelemetryWorker:
    """
    Non-blocking background telemetry worker.
    Uses an internal queue and persistent daemon thread to dispatch
    multipart/form-data payloads to the central server without impacting
    edge inspection loop or GPIO actuation latency.
    """

    def __init__(self, central_url: str = "http://192.168.137.1:8000", max_queue_size: int = 100):
        self.central_url = central_url.rstrip("/")
        self.endpoint = f"{self.central_url}/api/v1/inspections/run"
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.success_count = 0
        self.error_count = 0
        self.last_status: Optional[int] = None

        self.start()

    def start(self):
        """Starts the background sender thread."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="TelemetryWorker")
            self._thread.start()

    def _worker_loop(self):
        """Pulls items from the queue and sends HTTP POST requests."""
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._send_payload(item)
            except Exception as e:
                self.error_count += 1
                # Silent fail per architectural specification — edge sorting is already complete
                print(f"⚠️ [TELEMETRY] Delivery failed ({e})")
            finally:
                self._queue.task_done()

    def _send_payload(self, item: Dict[str, Any]):
        """Transmits snapshot JPEG and metadata to central server."""
        import json
        jpeg_bytes = item["jpeg_bytes"]
        fish_family = item.get("fish_family", "Scombridae")
        lot_id = item.get("lot_id")

        files = {
            "image": ("snapshot.jpg", jpeg_bytes, "image/jpeg"),
            "file": ("snapshot.jpg", jpeg_bytes, "image/jpeg"),
        }
        data = {
            "fish_family": fish_family,
            "family": fish_family
        }
        if lot_id:
            data["lot_id"] = lot_id
        if item.get("grade"):
            data["grade"] = item["grade"]
        if item.get("grade_confidence") is not None:
            data["grade_confidence"] = str(item["grade_confidence"])
        if item.get("defects") is not None:
            data["defects"] = json.dumps(item["defects"]) if isinstance(item["defects"], list) else str(item["defects"])
        if item.get("decision"):
            data["decision"] = item["decision"]
        if item.get("hardware_signal"):
            data["hardware_signal"] = item["hardware_signal"]
            data["conveyor_signal"] = item["hardware_signal"]
        if item.get("reason"):
            data["reason"] = item["reason"]
        if item.get("processing_time_ms") is not None:
            data["processing_time_ms"] = str(item["processing_time_ms"])

        try:
            resp = requests.post(self.endpoint, files=files, data=data, timeout=5.0)
            self.last_status = resp.status_code
            if resp.status_code in [200, 201]:
                self.success_count += 1
                print(f"📡 [TELEMETRY] Dispatched successfully -> {resp.status_code}")
            else:
                self.error_count += 1
                print(f"⚠️ [TELEMETRY] Central returned HTTP {resp.status_code}: {resp.text[:100]}")
        except requests.exceptions.RequestException as e:
            self.error_count += 1
            print(f"⚠️ [TELEMETRY] Network timeout / connection error: {e}")

    def enqueue(
        self,
        jpeg_bytes: bytes,
        fish_family: str = "Scombridae",
        lot_id: Optional[str] = None,
        grade: Optional[str] = None,
        grade_confidence: Optional[float] = None,
        defects: Optional[list] = None,
        decision: Optional[str] = None,
        hardware_signal: Optional[str] = None,
        reason: Optional[str] = None,
        processing_time_ms: Optional[int] = None
    ):
        """Enqueues an inspection record for asynchronous background delivery."""
        payload = {
            "jpeg_bytes": jpeg_bytes,
            "fish_family": fish_family,
            "lot_id": lot_id,
            "grade": grade,
            "grade_confidence": grade_confidence,
            "defects": defects,
            "decision": decision,
            "hardware_signal": hardware_signal,
            "reason": reason,
            "processing_time_ms": processing_time_ms,
            "timestamp": time.time()
        }
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            print("⚠️ [TELEMETRY] Queue full, dropping oldest event.")
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(payload)
            except Exception:
                pass
    def stop(self):
        """Stops the background worker thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


# Global singleton instance
_GLOBAL_TELEMETRY_WORKER: Optional[TelemetryWorker] = None


def get_telemetry_worker(central_url: str = "http://192.168.137.1:8000") -> TelemetryWorker:
    """Returns or creates the singleton TelemetryWorker."""
    global _GLOBAL_TELEMETRY_WORKER
    if _GLOBAL_TELEMETRY_WORKER is None:
        _GLOBAL_TELEMETRY_WORKER = TelemetryWorker(central_url=central_url)
    return _GLOBAL_TELEMETRY_WORKER


def dispatch_telemetry(
    jpeg_bytes: bytes,
    fish_family: str = "Scombridae",
    lot_id: Optional[str] = None,
    central_url: str = "http://192.168.137.1:8000"
):
    """Convenience function to dispatch inspection data asynchronously."""
    worker = get_telemetry_worker(central_url=central_url)
    worker.enqueue(jpeg_bytes=jpeg_bytes, fish_family=fish_family, lot_id=lot_id)
