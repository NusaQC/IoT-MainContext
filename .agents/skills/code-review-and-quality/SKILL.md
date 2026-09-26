---
name: code-review-and-quality
description: "Quality gate and multi-axis code review specifically tailored for NusaQC (COMPFEST 18 AIC - Smart Manufacturing). Covers correctness (SNI freshness, defect classes, zero escape), edge & IoT safety (3.3V GPIO, relay closed-loop), Next.js 16 + FastAPI architecture, performance (<150ms actuation), and eval compliance."
---

# Code Review & Quality Gate — NusaQC (COMPFEST 18 AIC)

Quality gate and multi-axis review framework for **NusaQC (AI-Powered Visual Quality Control & Digital Traceability System)** for COMPFEST 18 Artificial Intelligence Competition (AIC) - Smart Manufacturing Track.

---

## 1. Approval Philosophy

> **Approve code that delivers reliable, real-time quality control with sub-second closed-loop actuation and verifiable audit trails.** Reject speculative abstractions, unrequested dependencies, unsafe GPIO operations, and unverified data leakage in ML pipelines.

---

## 2. Five-Axis Review Criteria

### Axis 1: Correctness & AI Invariants
- **Model 1 (Freshness)**:
  - Output mapped correctly to SNI 2729:2013 standards: Grade A (Prima), Grade B (Segar/Domestik), Grade C (Afkir).
  - Softmax probabilities and threshold comparisons are mathematically sound.
  - **Zero Critical Escape Rule**: Grade C must NEVER be classified/passed as Grade A (`CER == 0.00%`).
- **Model 2 (Defects)**:
  - Exact 4 defect classes respected: `0: sisik_sisa`, `1: warna_abnormal`, `2: luka_robekan`, `3: lendir_berlebih`.
  - Coordinates properly converted between letterbox (640x640) and original frame dimensions.
  - NMS threshold (IoU 0.45, conf ≥ dynamic threshold) properly applied.
- **Decision Engine**:
  - Deterministic evaluation order: Grade C → FAIL; defects present → FAIL; Grade B low confidence → CONDITIONAL; Grade A/B high confidence → PASS.

### Axis 2: Edge Hardware & IoT Safety
- **3.3V Logic Protection**:
  - GPIO inputs (sensor E18-D80NK) must use internal pull-up (`GPIO.PUD_UP` / `Button(pull_up=True)`). Never expose GPIO pins to raw 5V.
  - Active LOW sensor triggers handled correctly (`fish_present = not GPIO.input(PIN)`).
- **Failsafe Actuation Logic**:
  - `FAIL` signal must IMMEDIATELY trigger relay cutoff (STOP motor conveyor) and siren/alarm.
  - Actuation happens synchronously and instantly (< 2 ms) upon decision.
- **Mock Hardware Fallback**:
  - `ENABLE_MOCK_HARDWARE=true` must always be runnable without `RPi.GPIO` installed (for jury demo & CI).
  - Mock controller must log colored ASCII terminal events without crashing.

### Axis 3: Architecture & Data Contracts
- **Backend API (FastAPI)**:
  - Returns dual-compatible payloads (snake_case + camelCase alias, e.g. `lotId` & `conveyorSignal`).
  - Thread-safe WebSocket broadcast with snapshot copy iteration (`list(self.active_connections)`).
  - Proper SQLite transaction scoping and SQLAlchemy session handling.
- **Telemetry Worker**:
  - HTTP telemetry dispatch from Edge to Central MUST run in a non-blocking daemon thread.
  - Failure to send telemetry must not block or crash local conveyor actuation.
- **Frontend (Next.js 16 + React 19)**:
  - Clear `"use client"` vs RSC boundaries.
  - WebSocket connection lifecycle properly cleaned up on unmount with reconnect backoff.
  - Responsive Tailwind CSS layouts, no layout shifts during bounding box overlays.

### Axis 4: Performance & Latency Budgets
- **Sub-Second Actuation Target**:
  - MobileNetV3 inference ≤ 30 ms.
  - YOLOv8s INT8 inference ~1.000 ms on RPi 4.
  - Combined actuation cycle ≤ 1.1s (critical for conveyor travel time).
- **ONNX Runtime Thread Tuning**:
  - Respect `NUSAQC_ORT_THREADS` (set to 4 on Cortex-A72 quad-core).
  - Offload ONNX calls to `run_in_executor` in async FastAPI/eval loops to prevent event loop starvation.

### Axis 5: Hackathon & Evaluation Integrity (Rulebook Compliance)
- **Anti-Leakage Validation**:
  - ML training data splits must use `GroupShuffleSplit` by session/day, NEVER naive random split.
- **Reproducibility**:
  - Evaluation metrics script runs via clean CLI (`python -m eval.run_all`).
  - Baseline and iteration results preserved in `eval/results/` for evaluation artifact generation.
- **Ponytail & Karpathy Discipline**:
  - Minimal diff: touch only relevant lines.
  - Reuse existing utilities in `lib/`, `services/`, and `hardware/`.
  - No orphaned imports or speculative future-proofing.

---

## 3. Pre-Commit / Code Freeze Checklist (20.00 WIB)

- [ ] Backend starts clean: `uv run fastapi dev app/main.py --port 8000`
- [ ] Frontend builds with zero TypeScript errors: `pnpm build`
- [ ] Mock Hardware Mode active & verified (`ENABLE_MOCK_HARDWARE=true`)
- [ ] ONNX weights resolvable or Simulation fallback works gracefully
- [ ] Git tags created according to schedule:
  - `checkpoint-1-baseline` (12.00 WIB)
  - `checkpoint-2-iteration` (15.00 WIB)
  - `checkpoint-3-integration` (18.00 WIB)
  - `final-submission` (20.00 WIB)
