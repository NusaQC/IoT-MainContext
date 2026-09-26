---
name: nusaqc-iot-async-stream
  Kontrak kode, konvensi, dan gotcha aktual untuk sistem NusaQC:
  Backend FastAPI, AI inference ONNX, GPIO hardware, async stream eval,
  WebSocket broadcast, dan frontend Next.js. Mencakup juga pola
  distributed edge-to-central (RPi → Central Laptop). Gunakan skill ini
  untuk semua pengerjaan di webdev/.
---

## Stack

| Layer | Teknologi | Path |
|---|---|---|
| Backend | FastAPI + SQLAlchemy + SQLite | `webdev/backend/` |
| AI Runtime | ONNX Runtime CPU | `backend/app/ai/` |
| Hardware | RPi.GPIO (real) / Mock | `backend/app/hardware/` |
| Frontend | Next.js 16 App Router + Tailwind | `webdev/frontend/` |
| Eval Suite | Python asyncio + psutil | `webdev/eval/` |

---

## 1. AI Inference Engine

**File:** `backend/app/ai/inference.py`

```python
engine = AIInferenceEngine()

# Model 1 — MobileNetV3-Small (224×224 RGB)
# Input name: 'x', Output: 'linear_1' shape (3,) → softmax → Grade A/B/C
result = engine.predict_freshness(pil_image)
# → {"grade": "A"|"B"|"C", "confidence": float}

# Model 2 — YOLOv8s ONNX (640×640 letterbox)
# Input name: 'images', Output: 'output0' shape (1, channels, 8400)
# → transpose (8400, ch) → NMS via cv2.dnn.NMSBoxes → unletterbox coords
defects = engine.predict_defects(pil_image, confidence_threshold=0.55, iou_threshold=0.45)
# → [{"label": str, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
```

**Defect classes:** `{0: "sisik_sisa", 1: "warna_abnormal", 2: "luka_robekan", 3: "lendir_berlebih"}`

**Fallback:** Jika ONNX weights tidak ada → Simulation Mode (return dummy A / []).
Model weights dicari di `settings.MODEL_DIR` dengan nama:
- `mobilenetv3_freshness.onnx`
- `nusaqc_model2_defect_detector.onnx`

**Thread-safety:** ONNX sessions CPU thread-safe — aman dipakai dari banyak worker thread sekaligus.

---

## 2. Decision Engine

**File:** `backend/app/services/decision_engine.py`

```python
decision, signal, reason = DecisionEngine.evaluate(
    grade, grade_confidence, defects, confidence_threshold=0.75
)
# Returns: ("PASS"|"CONDITIONAL"|"FAIL", "GREEN"|"YELLOW"|"RED", str)
```

**Prioritas aturan (deterministik, urutan penting):**
1. Grade C → **FAIL / RED**
2. `len(defects) > 0` → **FAIL / RED**
3. Grade B + `confidence < threshold` → **CONDITIONAL / YELLOW**
4. Grade A atau B (confidence ≥ threshold) → **PASS / GREEN**

`confidence_threshold` dibaca dari `SystemSetting.key="global_config"` di SQLite, default `0.75`.

---

## 3. Inspection Pipeline (end-to-end)

**File:** `backend/app/services/inspection_service.py`

Urutan langkah dalam `InspectionService.process_inspection()`:
1. Buat Lot ID (`LOT-YYYYMMDD-NNN`)
2. Simpan JPEG ke `UPLOAD_DIR`
3. `engine.predict_freshness()` → grade
4. Baca `confidence_threshold` dari DB
5. `engine.predict_defects()` → filter by threshold
6. `DecisionEngine.evaluate()` → decision + signal
7. `hardware_controller.trigger_signal(signal)` (sync, non-blocking)
8. Persist ke SQLite `inspections` table
9. `await ws_manager.broadcast_json({"event": "NEW_INSPECTION", "data": result.dict()})`

**Gotcha:** Step 9 pakai `await` → `process_inspection` harus `async`. Hardware actuation (step 7) adalah sync call — tidak perlu await.

---

## 4. WebSocket

**File:** `backend/app/core/websocket.py`
**Endpoint:** `ws://localhost:8000/ws/events`

**Broadcast payload format** (yang diterima frontend):
```json
{"event": "NEW_INSPECTION", "data": { ...InspectionResult fields... }}
```

**Pattern aman** (sudah difix — jangan ubah balik):
```python
async def broadcast_json(self, data: dict):
    dead = []
    for conn in list(self.active_connections):  # snapshot dulu
        try:
            await conn.send_json(data)
        except Exception:
            dead.append(conn)
    for conn in dead:
        self.disconnect(conn)
```

**Frontend hook:** `frontend/lib/useWebSocket.ts` — `useInspectionWebSocket(onNewInspection)`
- Auto-reconnect setiap 3 detik jika disconnect
- Parse `payload.event === "NEW_INSPECTION"` → panggil callback

---

## 5. GPIO Hardware

**File:** `backend/app/hardware/gpio_controller.py`

**Pinout BCM Raspberry Pi 4** (sudah difix — single source of truth):
```python
PIN_IR_SENSOR      = 17  # INPUT, Active LOW, pull-up internal 3.3V
PIN_BUZZER         = 18  # OUTPUT, PWM
PIN_LIGHT_GREEN    = 27  # OUTPUT, 220Ω series
PIN_LIGHT_YELLOW   = 22  # OUTPUT, 220Ω series
PIN_LIGHT_RED      = 23  # OUTPUT, 220Ω series
PIN_CONVEYOR_RELAY = 25  # OUTPUT, Active HIGH
```

**Baca sensor IR (edge code, bukan backend):**
```python
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
fish_present = not GPIO.input(17)  # True saat sensor aktif (LOW)
```

**Sinyal → aksi GPIO:**
| Signal | Relay 25 | LED | Buzzer |
|--------|----------|-----|--------|
| GREEN  | HIGH (motor jalan) | 27 ON | OFF |
| YELLOW | HIGH | 22 ON | HIGH (bip) |
| RED    | LOW (motor STOP) | 23 ON | HIGH (alarm) |

**Mode switching** via `ENABLE_MOCK_HARDWARE` env var:
- `true` → `MockHardwareController` (terminal log berwarna, no GPIO import)
- `false` → `GPIOController` (RPi.GPIO fisik)

**File:** `backend/app/hardware/__init__.py` — `get_hardware_controller()` singleton.

---

## 6. Async Stream Eval

**File:** `eval/test_async_stream.py`

**Pola wajib — satu `asyncio.run()` per eksekusi:**
```python
def run_async_stream_evaluation(save_results=True) -> dict:
    async def _run_all():
        # semua await di sini — burst + 3× endurance
        burst = await simulate_async_conveyor_queue(engine, frames, max_workers=2)
        for _ in range(3):
            await simulate_async_conveyor_queue(engine, frames[:15], max_workers=2)
        return build_payload(burst, ...)
    return asyncio.run(_run_all())
```

**Jangan** pakai multiple `asyncio.run()` berurutan — crash jika dipanggil dari pytest-asyncio atau async context.

**Worker queue pattern:** `run_in_executor(None, engine.predict_freshness, frame)` — ONNX inference adalah blocking CPU, harus dioffload ke thread pool agar tidak memblokir event loop.

**Jalankan satu suite:**
```bash
cd webdev
python -m eval.run_all --suite 5   # async stream saja
python -m eval.run_all             # semua suite
```

**Hasil disimpan ke:** `eval/results/async_stream_results.json`

---

## 7. TypeScript Types & API Client

**File:** `frontend/types/index.ts` — semua tipe pakai dual snake_case + camelCase:
```ts
type Decision = "PASS" | "FAIL" | "CONDITIONAL"
type HardwareSignal = "GREEN" | "YELLOW" | "RED"
type Grade = "A" | "B" | "C"
type DefectLabel = "sisik_sisa" | "warna_abnormal" | "luka_robekan" | "lendir_berlebih" | string

interface InspectionResult { lotId, grade, confidence, defects: Defect[], decision, conveyorSignal, processingTimeMs, imageUrl, ... }
interface LotRecord { id, lotId, grade, decision, conveyorSignal, defectsCount, storageSlot, dispatchId, ... }
interface StorageSlot { slot_id, zone: "cold"|"frozen", lot_id, lot?: LotRecord }
interface DispatchRecord { dispatch_id, buyer_name, destination, container_no, status, lots?: LotRecord[] }
```

**File:** `frontend/lib/api.ts` — `API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"`

Key functions:
```ts
runInspection(imageFile, fishFamily, lotId?)        // POST /api/v1/inspections/run
fetchDashboardStats()                               // GET /api/v1/dashboard/stats
assignStorageSlot(slotId, lotId)                    // POST /api/v1/storage/assign
createDispatch({ buyer_name, destination, lot_ids }) // POST /api/v1/dispatch
updateDispatchStatus(dispatchId, status)
overrideLotDecision(lotId, decision, reason)
```

---

## 8. Konvensi Response Backend

Backend selalu return **dual-compatible payload** (snake_case + camelCase alias):
```python
# Di endpoint handler, tambahkan alias setelah model_dump():
res_dict = result.model_dump()
res_dict.update({
    "lotId": result.lot_id,
    "conveyorSignal": result.hardware_signal,
    "processingTimeMs": result.processing_time_ms,
    ...
})
return res_dict
```

Ini konvensi wajib — frontend mengkonsumsi keduanya.

---

## 9. Eval Config — Path Resolution

**File:** `eval/config.py` — STRICT: raise `FileNotFoundError` jika file tidak ada.

```python
DATASET_FRESHNESS_DIR = Path(r"d:\main\Documents\...\datasets\mobilenet")
FRESHNESS_MODEL_PATH  = MODEL_DIR / "mobilenetv3_freshness.onnx"
DEFECT_MODEL_PATH     = MODEL_DIR / "nusaqc_model2_defect_detector.onnx"
```

Jika menjalankan eval dari luar `webdev/`, pastikan `sys.path` include `webdev/` dan `webdev/backend/`.

---

## 10. Gotcha & Risiko

| # | Masalah | Lokasi | Status |
|---|---------|--------|--------|
| 1 | GPIO pin mapping salah (relay=17, buzzer=24) | `gpio_controller.py` | ✅ Fixed |
| 2 | `broadcast_json` iterasi list sambil dihapus | `websocket.py` | ✅ Fixed |
| 3 | Multiple `asyncio.run()` crash di async context | `test_async_stream.py` | ✅ Fixed |
| 4 | `useInspectionWebSocket` punya stale closure risk jika `onNewInspection` tidak distabilkan dengan `useCallback` | `useWebSocket.ts` | ⚠️ Perlu cek |
| 5 | `get_status()` di `GPIOController` return hardcoded "ACTIVE"/"GREEN" tanpa baca state pin aktual | `gpio_controller.py` | ⚠️ Belum fix |

---

## 11. Distributed Edge-to-Central — Pola Integrasi

Edge node (RPi 4) dan Central Server (Laptop) terhubung via Mobile Hotspot:
- **Edge IP:** `192.168.137.251` (RPi, user: `dti`)
- **Central IP:** `192.168.137.1` (Laptop, gateway hotspot)

### Telemetry POST dari Edge ke Central
```python
# Dijalankan di background thread — JANGAN await, JANGAN blokir main loop
requests.post(
    "http://192.168.137.1:8000/api/v1/inspections/run",
    files={"file": ("snap.jpg", jpeg_bytes, "image/jpeg")},
    data={"fish_family": "Scombridae", "lot_id": "LOT-20260926-001"},
    timeout=5
)
```

### MJPEG Stream (Edge Camera Live)
- Edge server: `http://192.168.137.251:8080/stream`
- Health check: `http://192.168.137.251:8080/health` → `{"status":"ok","camera_active":true}`
- Frontend embed: `<img src="http://192.168.137.251:8080/stream" />` atau via Next.js proxy

### Alur Data End-to-End
```
E18 Sensor → GPIO 17 → capture frame → ONNX dual inference
→ Decision Engine → GPIO actuate (instant, sync)
→ Background thread → POST Central → SQLite → WebSocket → Dashboard
```

**Constraint penting:** Aktuasi GPIO fisik HARUS selesai sebelum telemetry dikirim — urutan ini tidak boleh dibalik. Latensi aktuasi lokal ≤ 1.1 detik; telemetry bersifat best-effort (timeout 5 detik, silent fail).

---

## 12. Stack Referensi Skill Terkait

| Area | Skill |
|---|---|
| ML Pipeline, ONNX, INT8, metrik AI | `nusaqc-ai-ml-pipeline` |
| RPi4 GPIO wiring, sensor, edge deploy | `nusaqc-iot-edge-deployment` |
| Test suite, eval artifact, checkpoint | `nusaqc-eval-track` |
| Kode webdev (skill ini) | `nusaqc-iot-async-stream` |
