---
name: nusaqc-iot-edge-deployment
description: >
  Cetak biru deployment IoT NusaQC pada Raspberry Pi 4 (edge node):
  arsitektur distributed edge-to-central, GPIO pinout aktual, sensor
  E18-D80NK, MJPEG stream server, telemetry worker, jaringan hotspot
  venue, dan prosedur kontinjensi mock mode. Gunakan skill ini untuk
  semua pekerjaan di edge/ dan deployment RPi.
---

## 1. Topologi Sistem: Distributed Edge-to-Central

```
EDGE NODE (Raspberry Pi 4 — 192.168.137.251)
├── USB Webcam UVC (Overhead Snapshot + MJPEG :8080)
├── Sensor IR E18-D80NK (GPIO 17, Active LOW)
├── LED Traffic Light (GPIO 27/22/23)
├── Active Buzzer (GPIO 18, PWM)
├── Relay Konveyor (GPIO 25, Active HIGH)
├── Dual ONNX Inference (MobileNetV3 + YOLOv8s INT8)
└── Background Telemetry Worker (non-blocking thread)
         │
         │ HTTP POST multipart/form-data
         │ JPEG snapshot + metadata
         ▼
CENTRAL SERVER (Laptop Tim — 192.168.137.1)
├── FastAPI Backend :8000
├── SQLite Database
├── Next.js Frontend :3000
└── WebSocket Broadcast → Operator Dashboard
```

---

## 2. Pinout BCM Raspberry Pi 4 (Single Source of Truth)

| Pin Fisik | BCM | Perangkat | Keterangan |
|---|---|---|---|
| Pin 02 | 5V Power | Sensor E18 VCC + Relay VCC | Rail 5V paralel |
| Pin 04 | 5V Power | Cooling Fan + | — |
| Pin 06 | GND | Cooling Fan - | — |
| Pin 09 | GND | Katoda ketiga LED | LED Hijau, Kuning, Merah |
| **Pin 11** | **GPIO 17** | **Sensor E18-D80NK (Signal)** | Active LOW, internal pull-up |
| **Pin 12** | **GPIO 18** | **Active Buzzer** | PWM output |
| **Pin 13** | **GPIO 27** | **LED Hijau (PASS)** | 220Ω seri |
| Pin 14 | GND | Sensor E18 GND | Kabel biru sensor |
| **Pin 15** | **GPIO 22** | **LED Kuning (CONDITIONAL)** | 220Ω seri |
| **Pin 16** | **GPIO 23** | **LED Merah (FAIL)** | 220Ω seri |
| Pin 20 | GND | Relay GND + Buzzer GND | — |
| **Pin 22** | **GPIO 25** | **Relay Konveyor (IN)** | Active HIGH |
| USB 3.0 | — | USB Webcam | Port biru |

**Resistor LED:** $I = \frac{3.3V - 2.0V}{220\Omega} \approx 5.9\text{ mA}$ (aman, max 16 mA/pin)

### Logika Sinyal → GPIO
| Signal | Relay 25 | LED ON | Buzzer |
|---|---|---|---|
| GREEN (PASS) | HIGH (motor jalan) | GPIO 27 | OFF |
| YELLOW (CONDITIONAL) | HIGH | GPIO 22 | HIGH (bip singkat) |
| RED (FAIL) | LOW (**motor STOP**) | GPIO 23 | HIGH (alarm kontinyu) |

---

## 3. Sensor E18-D80NK — Wiring & Kalibrasi

### Proteksi Tegangan 3.3V — KRITIS
> ⚠️ GPIO RPi TIDAK 5V-TOLERANT. Sensor butuh 5V untuk pemancar IR-nya, tapi output sinyal (kabel Hitam) adalah NPN Open-Collector.

**Solusi aman tanpa voltage divider:**
```python
# gpiozero (direkomendasikan untuk edge)
from gpiozero import Button
sensor = Button(17, pull_up=True)
# pin idle → 3.3V (internal pull-up); saat deteksi → ditarik ke GND (0V) — aman

# RPi.GPIO (alternatif)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
fish_present = not GPIO.input(17)  # True saat LOW (ikan terdeteksi)
```

### Kalibrasi Jarak di Venue
1. Hubungkan sensor (Cokelat=5V, Biru=GND, Hitam=GPIO17)
2. Meja kosong → putar potensiometer CCW sampai LED indikator sensor **PADAM**
3. Letakkan ikan → putar CW sampai LED **MENYALA TEGAS**
4. Verifikasi: `fish_present = True` saat ikan, `False` saat kosong

---

## 4. Alur Closed-Loop Sub-Detik (< 150 ms total)

```
[1] E18-D80NK → GPIO 17 LOW → trigger akuisisi
[2] OpenCV cv2.VideoCapture → 1 frame dari buffer
[3] Model 1 (MobileNetV3) → Grade A/B/C (~28 ms)
[4] Model 2 (YOLOv8s INT8) → BBoxes cacat (~1.000 ms)
[5] Decision Engine → PASS/CONDITIONAL/FAIL
[6] GPIO actuate instantly (sync, < 1 ms)
    PASS:  GPIO 27 ON, Relay 25 HIGH
    COND:  GPIO 22 ON, Buzzer bip
    FAIL:  GPIO 23 ON, Relay 25 LOW (STOP), Alarm
[7] Background thread → POST ke Central Server (non-blocking)
```

**Penting:** Langkah 1–6 berjalan di main thread sinkron. Step 7 harus di thread terpisah agar tidak menambah latensi aktuasi.

---

## 5. Telemetry Worker Pattern (Non-Blocking)

```python
import threading, requests

def _send_telemetry(snapshot_bytes, fish_family, lot_id, central_url):
    try:
        requests.post(
            f"{central_url}/api/v1/inspections/run",
            files={"file": ("snapshot.jpg", snapshot_bytes, "image/jpeg")},
            data={"fish_family": fish_family, "lot_id": lot_id},
            timeout=5
        )
    except Exception:
        pass  # Silent fail — aktuasi lokal sudah selesai, log tidak kritis

def dispatch_telemetry(snapshot_bytes, fish_family, lot_id, central_url):
    t = threading.Thread(
        target=_send_telemetry,
        args=(snapshot_bytes, fish_family, lot_id, central_url),
        daemon=True
    )
    t.start()
```

---

## 6. MJPEG Stream Server (:8080)

```python
# edge/mjpeg_server.py
# Endpoint: http://192.168.137.251:8080/stream
# Health: http://192.168.137.251:8080/health → {"status":"ok","camera_active":true}

# Frontend mengkonsumsi sebagai <img src="http://192.168.137.251:8080/stream" />
# atau via Next.js proxy untuk CORS
```

---

## 7. Deployment di Venue Hackathon

### Transfer Files ke RPi
```bash
scp -r edge run.sh dti@192.168.137.251:~/nusaqc/
scp -r AI/model-1/ AI/model-2/ dti@192.168.137.251:~/nusaqc/AI/
```

### Jalankan Edge Node
```bash
ssh dti@192.168.137.251
cd /home/dti/nusaqc
python3 -m venv venv && source venv/bin/activate
pip install -r edge/requirements-edge.txt
python -m edge.main --cam 0 --mjpeg-port 8080 --central-url http://192.168.137.1:8000
```

### Verifikasi
```bash
# Di browser laptop:
http://192.168.137.251:8080/health   # → {"status":"ok","camera_active":true}
http://localhost:3000/inspection      # Widget "Live Edge Camera" harus hijau LIVE
```

---

## 8. Skema Jaringan Venue

```
Internet → Router Venue → [WiFi] → Laptop Tim (Windows)
                                         │
                                   Mobile Hotspot
                                   192.168.137.1
                                         │ [WiFi / USB-C tether]
                                   Raspberry Pi 4
                                   192.168.137.251
                                   user: dti
                                   hostname: raspberrypi.local
```

### Troubleshooting Jaringan
```bash
# RPi tidak respond di raspberrypi.local:
# Cek di Windows: Settings → Network & Internet → Mobile Hotspot → Connected devices
ssh dti@<IP_DARI_HOTSPOT_PANEL>

# Ganti SSID WiFi di RPi:
sudo nmcli dev wifi connect "<SSID_BARU>" password "<PASSWORD>"
```

---

## 9. Kontinjensi Mock Mode (Fallback Saat Hardware Gagal)

Sesuai Rulebook COMPFEST 18 AIC Bagian 2.6 Poin 5:

| Skenario | Fallback |
|---|---|
| GPIO/sensor mati | `ENABLE_MOCK_HARDWARE=true` di backend — log terminal berwarna ASCII |
| Kamera RPi tidak aktif | Toggle frontend ke mode **"Upload Berkas"** atau **"Kamera Laptop"** |
| RPi offline total | Backend FastAPI di laptop otomatis handle inference lokal (ONNX CPU) |
| Model weights hilang | Simulation Mode — dummy output, dashboard tetap jalan |

**Fitur inti yang dijamin tanpa hardware fisik:**
- Deteksi kesegaran + bounding box defek
- Visualisasi lot storage map
- Export manifes dispatch CSV

---

## 10. Latensi Empiris RPi 4 (1.5 GHz Cortex-A72)

| Komponen | 2 Thread | 4 Thread | Rekomendasi |
|---|---|---|---|
| Model 1 FP32 | 28.3 ms | 33.0 ms | **2 thread** (lebih cepat) |
| Model 2 FP32 | 2.440 ms | 1.849 ms | 4 thread baseline |
| Model 2 INT8 | — | ~950–1.100 ms | **4 thread + INT8** ✅ |
| Total Dual-AI | 2.469 ms | — | **~1.028 ms (INT8)** ✅ |

Set: `NUSAQC_ORT_THREADS=4` (env var ke InferenceSession options)
