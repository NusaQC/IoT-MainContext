"""
NusaQC Edge Node Package
Distributed Edge-to-Central Quality Control for Smart Fish Processing.
"""

from edge.camera import CameraManager
from edge.gpio import GPIOController, get_gpio_controller
from edge.inference import EdgeInferenceEngine
from edge.decision import DecisionEngine
from edge.telemetry import TelemetryWorker, dispatch_telemetry
from edge.mjpeg_server import start_mjpeg_server, stop_mjpeg_server

__version__ = "1.0.0"

__all__ = [
    "CameraManager",
    "GPIOController",
    "get_gpio_controller",
    "EdgeInferenceEngine",
    "DecisionEngine",
    "TelemetryWorker",
    "dispatch_telemetry",
    "start_mjpeg_server",
    "stop_mjpeg_server",
]
