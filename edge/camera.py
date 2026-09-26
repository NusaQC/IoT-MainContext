import cv2
import time
import threading
import numpy as np
from typing import Optional, Generator, Tuple


class CameraManager:
    """
    Thread-safe OpenCV camera manager with background frame buffer draining.
    Prevents VideoCapture internal buffer lag and provides instant latest frames,
    snapshots, and MJPEG multipart generator. Includes synthetic frame fallback.
    """

    def __init__(self, camera_source: int | str = 0, width: int = 1280, height: int = 720, target_fps: int = 20):
        self.camera_source = camera_source
        self.target_width = width
        self.target_height = height
        self.target_fps = target_fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._is_mock = False
        self._frame_count = 0
        self._last_frame_time = 0.0

        self._init_camera()
        self._start_capture_thread()

    def _init_camera(self) -> None:
        """Attempts to open the physical camera; falls back to synthetic mock if unavailable."""
        try:
            # Handle int string if passed
            source = int(self.camera_source) if str(self.camera_source).isdigit() else self.camera_source
            cap = cv2.VideoCapture(source)

            # Suggest resolutions & format for USB UVC webcams
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Test-read one frame
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                self._cap = cap
                self._is_mock = False
                print(f"📷 [CAMERA] Physical camera opened on '{self.camera_source}' ({frame.shape[1]}x{frame.shape[0]})")
                with self._lock:
                    self._latest_frame = frame
                return
            else:
                if cap.isOpened():
                    cap.release()
                print(f"⚠️ [CAMERA] Physical camera '{self.camera_source}' failed to return frames. Activating Mock Camera Mode.")
        except Exception as e:
            print(f"⚠️ [CAMERA] Camera init error ({e}). Activating Mock Camera Mode.")

        self._cap = None
        self._is_mock = True
        self._generate_mock_frame()

    def _generate_mock_frame(self) -> np.ndarray:
        """Generates a realistic synthetic test pattern resembling a conveyor inspection zone."""
        img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        # Background: dark slate conveyor belt
        img[:] = (35, 38, 42)

        # Conveyor guide rails
        cv2.line(img, (0, 80), (self.target_width, 80), (70, 75, 80), 3)
        cv2.line(img, (0, self.target_height - 80), (self.target_width, self.target_height - 80), (70, 75, 80), 3)

        # Conveyor belt track markings
        t = time.time()
        offset = int((t * 120) % 100)
        for x in range(-100 + offset, self.target_width + 100, 100):
            cv2.line(img, (x, 85), (x + 30, self.target_height - 85), (48, 52, 58), 2)

        # Center Inspection zone box
        cx, cy = self.target_width // 2, self.target_height // 2
        box_w, box_h = 480, 260
        x1, y1 = cx - box_w // 2, cy - box_h // 2
        x2, y2 = cx + box_w // 2, cy + box_h // 2
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 220), 2)

        # Synthetic Fish Silhouette in the center
        # Ellipse body
        cv2.ellipse(img, (cx, cy), (160, 55), 0, 0, 360, (140, 150, 160), -1)
        # Tail
        tail_pts = np.array([[cx - 150, cy], [cx - 210, cy - 40], [cx - 210, cy + 40]], np.int32)
        cv2.fillPoly(img, [tail_pts], (120, 130, 140))
        # Eye
        cv2.circle(img, (cx + 110, cy - 12), 8, (20, 20, 20), -1)
        cv2.circle(img, (cx + 112, cy - 14), 2, (240, 240, 240), -1)

        # Status text overlay
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, f"NusaQC Edge Cam [MOCK SIMULATION] - {time_str}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"CONVEYOR ACTIVE | IR SENSOR READY (BCM 17)", (20, self.target_height - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 180), 2, cv2.LINE_AA)

        with self._lock:
            self._latest_frame = img
            self._last_frame_time = t
        return img

    def _start_capture_thread(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="EdgeCameraThread")
        self._thread.start()

    def _capture_loop(self) -> None:
        """Background thread continuously grabbing frames to ensure zero buffer latency."""
        interval = 1.0 / max(1, self.target_fps)
        while self._running:
            start_t = time.time()
            if self._cap is not None and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                        self._frame_count += 1
                        self._last_frame_time = time.time()
                else:
                    # Temporary read glitch, fallback
                    time.sleep(0.02)
            else:
                self._generate_mock_frame()
                self._frame_count += 1

            elapsed = time.time() - start_t
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_frame(self) -> Optional[np.ndarray]:
        """Returns the latest captured BGR frame safely copied from the buffer."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def get_snapshot_jpeg(self, quality: int = 85) -> Optional[bytes]:
        """Encodes the latest frame as JPEG bytes."""
        frame = self.get_frame()
        if frame is None:
            return None
        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if success:
            return encoded.tobytes()
        return None

    def is_camera_active(self) -> bool:
        """Returns True if the camera is functioning (real or active mock)."""
        return self._latest_frame is not None and (time.time() - self._last_frame_time) < 2.0

    def is_mock(self) -> bool:
        """Returns True if running in Mock Camera Mode."""
        return self._is_mock

    def generate_mjpeg(self, target_fps: int = 15) -> Generator[bytes, None, None]:
        """
        Generator yielding HTTP multipart MJPEG chunks:
        --frame\\r\\nContent-Type: image/jpeg\\r\\n\\r\\n[BYTES]\\r\\n
        """
        interval = 1.0 / max(1, target_fps)
        while self._running:
            t0 = time.time()
            frame = self.get_frame()
            if frame is not None:
                success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if success:
                    jpeg_bytes = encoded.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg_bytes)).encode("ascii") + b"\r\n\r\n" +
                        jpeg_bytes +
                        b"\r\n"
                    )
            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def release(self) -> None:
        """Stops the capture thread and releases camera hardware."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        print("📷 [CAMERA] Camera pipeline successfully stopped.")
