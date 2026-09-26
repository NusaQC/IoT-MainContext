import os
import sys
import time
import signal
import argparse
from typing import Optional

from edge.camera import CameraManager
from edge.gpio import get_gpio_controller
from edge.inference import EdgeInferenceEngine
from edge.decision import DecisionEngine
from edge.telemetry import TelemetryWorker
from edge.mjpeg_server import start_mjpeg_server, stop_mjpeg_server


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="NusaQC Edge Node Orchestrator (Raspberry Pi 4 / Live Sortasi)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cam", type=str, default="0",
                        help="Camera index or V4L2 device path (e.g. 0 or /dev/video0)")
    parser.add_argument("--mjpeg-host", type=str, default="0.0.0.0",
                        help="Host IP address to bind MJPEG streaming server")
    parser.add_argument("--mjpeg-port", type=int, default=8080,
                        help="HTTP port for MJPEG stream and health endpoints")
    parser.add_argument("--central-url", type=str, default="http://192.168.137.1:8000",
                        help="Base URL of central FastAPI backend server")
    parser.add_argument("--models-dir", type=str, default="AI",
                        help="Directory containing ONNX model weights (model-1 and model-2)")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("NUSAQC_ORT_THREADS", 4)),
                        help="CPU intra-op threads for ONNX Runtime")
    parser.add_argument("--confidence-thresh", type=float, default=0.75,
                        help="Confidence threshold for Grade B secondary inspection")
    parser.add_argument("--defect-conf-thresh", type=float, default=0.50,
                        help="Confidence threshold for YOLOv8 surface defect detections")
    parser.add_argument("--family", type=str, default="Scombridae",
                        help="Default fish biological family classification")
    parser.add_argument("--simulate-trigger", type=float, default=0.0,
                        help="If > 0, automatically simulate fish trigger every N seconds (for testing/demo)")
    parser.add_argument("--mock-hardware", action="store_true",
                        help="Force mock GPIO mode regardless of hardware platform")
    parser.add_argument("--auto-resume-seconds", type=float, default=5.0,
                        help="Seconds before auto-resetting conveyor after FAIL/CONDITIONAL actuation")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 65)
    print(" 🐟 NusaQC Edge Node — Closed-Loop Sorting & Telemetry System")
    print("=" * 65)
    print(f" • Camera Source     : {args.cam}")
    print(f" • MJPEG Server      : http://{args.mjpeg_host}:{args.mjpeg_port}/stream")
    print(f" • Central Server    : {args.central_url}")
    print(f" • CPU ORT Threads   : {args.threads}")
    print(f" • Models Directory  : {args.models_dir}")
    print(f" • Mock Hardware     : {args.mock_hardware}")
    print("=" * 65)

    # 1. Initialize Camera
    print("📸 Initializing Camera Subsystem...")
    cam_source = int(args.cam) if args.cam.isdigit() else args.cam
    camera = CameraManager(camera_source=cam_source, width=1280, height=720, target_fps=20)

    # 2. Initialize GPIO Controller
    print("⚡ Initializing GPIO Hardware Subsystem...")
    gpio = get_gpio_controller(force_mock=args.mock_hardware)

    # 3. Start MJPEG HTTP Server
    print("🌐 Starting MJPEG Streaming & Health Service...")
    start_mjpeg_server(camera=camera, gpio_controller=gpio, host=args.mjpeg_host, port=args.mjpeg_port)

    # 4. Initialize Local Dual-AI Inference Engine
    print("🧠 Initializing Local Dual-AI Inference Sessions...")
    inference_engine = EdgeInferenceEngine(models_dir=args.models_dir, num_threads=args.threads)

    # 5. Initialize Background Telemetry Dispatcher
    print("📡 Initializing Telemetry Worker...")
    telemetry = TelemetryWorker(central_url=args.central_url)

    # Signal Handling for Graceful Shutdown
    shutdown_requested = False

    def handle_sigint(sig, frame):
        nonlocal shutdown_requested
        print("\n🛑 Shutdown signal received. Cleaning up...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    # Ensure Initial Conveyor Normal State
    gpio.trigger_signal("GREEN")

    # State Variables for Edge Loop
    last_trigger_time = 0.0
    min_trigger_interval = 1.2  # Seconds minimum between consecutive inspections
    last_simulate_time = time.time()
    last_actuation_time = 0.0
    pending_resume = False
    inspection_counter = 0

    print("\n✅ NusaQC Edge Node is running and waiting for conveyor sensor triggers.")
    print("   [Trigger: GPIO 17 Active LOW (E18-D80NK) or simulation timer]\n")

    try:
        while not shutdown_requested:
            current_time = time.time()

            # Check if auto-resume needed after a FAIL/CONDITIONAL stop
            if pending_resume and (current_time - last_actuation_time) >= args.auto_resume_seconds:
                print("🔄 Auto-resuming conveyor to GREEN (Normal Sorting)...")
                gpio.trigger_signal("GREEN")
                pending_resume = False

            # Check for trigger condition
            sensor_triggered = False

            # A. Physical IR Sensor check (Active LOW)
            if gpio.is_fish_present():
                if (current_time - last_trigger_time) >= min_trigger_interval:
                    sensor_triggered = True
                    print(f"⚡ [SENSOR] IR Sensor triggered on GPIO 17! Object detected.")

            # B. Simulated trigger timer (if configured)
            if not sensor_triggered and args.simulate_trigger > 0:
                if (current_time - last_simulate_time) >= args.simulate_trigger:
                    sensor_triggered = True
                    last_simulate_time = current_time
                    print(f"⏱️ [SIMULATION] Auto-trigger interval reached ({args.simulate_trigger}s).")

            if sensor_triggered:
                last_trigger_time = current_time
                inspection_counter += 1
                lot_id = f"LOT-{time.strftime('%Y%m%d')}-{inspection_counter:04d}"

                print(f"\n--- [INSPECTION #{inspection_counter} | {lot_id}] ---")

                # Step 1: Instant snapshot from camera buffer
                t_snap_start = time.time()
                frame = camera.get_frame()
                if frame is None:
                    print("⚠️ Failed to capture frame from buffer. Skipping inspection.")
                    continue
                snap_latency_ms = (time.time() - t_snap_start) * 1000.0

                # Step 2: Local Dual ONNX Inference
                t_inf_start = time.time()
                dual_result = inference_engine.predict_dual(
                    bgr_img=frame,
                    defect_confidence_threshold=args.defect_conf_thresh
                )
                inf_latency_ms = (time.time() - t_inf_start) * 1000.0

                grade = dual_result["grade"]
                grade_conf = dual_result["grade_confidence"]
                defects = dual_result["defects"]

                # Step 3: Decision Engine Evaluation
                decision, signal_out, reason = DecisionEngine.evaluate(
                    grade=grade,
                    grade_confidence=grade_conf,
                    defects=defects,
                    confidence_threshold=args.confidence_thresh
                )

                # Step 4: INSTANT GPIO ACTUATION (Synchronous, Sub-Detik)
                t_act_start = time.time()
                gpio.trigger_signal(signal_out)
                act_latency_ms = (time.time() - t_act_start) * 1000.0
                last_actuation_time = current_time

                if decision in ["FAIL", "CONDITIONAL"]:
                    pending_resume = True

                total_local_loop_ms = (time.time() - t_snap_start) * 1000.0

                print(f"📊 Result: Decision={decision} | Signal={signal_out} | Grade={grade} ({int(grade_conf*100)}%) | Defects={len(defects)}")
                print(f"📝 Reason: {reason}")
                print(f"⏱️ Latency: Snap={snap_latency_ms:.1f}ms | M1={dual_result['latency']['freshness_ms']:.1f}ms | M2={dual_result['latency']['defect_ms']:.1f}ms | GPIO={act_latency_ms:.1f}ms | Total={total_local_loop_ms:.1f}ms")

                # Step 5: NON-BLOCKING TELEMETRY DISPATCH (Background Worker)
                try:
                    annotated_frame = EdgeInferenceEngine.annotate_frame(
                        bgr_img=frame,
                        grade=grade,
                        confidence=grade_conf,
                        defects=defects,
                        decision=decision,
                        total_ms=total_local_loop_ms
                    )
                    import cv2
                    _, enc = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    jpeg_bytes = enc.tobytes()

                    telemetry.enqueue(
                        jpeg_bytes=jpeg_bytes,
                        fish_family=args.family,
                        lot_id=lot_id,
                        grade=grade,
                        grade_confidence=grade_conf,
                        defects=defects,
                        decision=decision,
                        hardware_signal=signal_out,
                        reason=reason,
                        processing_time_ms=int(total_local_loop_ms)
                    )
                except Exception as e:
                    print(f"⚠️ Telemetry dispatch error: {e}")

                # Wait for IR beam to clear before next iteration to avoid debounce re-trigger
                debounce_wait = 0.0
                while gpio.is_fish_present() and debounce_wait < 1.0:
                    time.sleep(0.05)
                    debounce_wait += 0.05

            time.sleep(0.02)  # 50 Hz polling rate for IR sensor responsiveness

    finally:
        print("\n🧹 Releasing resources and shutting down...")
        gpio.cleanup()
        stop_mjpeg_server()
        camera.release()
        telemetry.stop()
        print("🏁 NusaQC Edge Node shutdown complete.")


if __name__ == "__main__":
    main()
