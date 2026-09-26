# NusaQC — Rencana & Panduan Implementasi IoT & Edge Hardware
### COMPFEST 18 AI Innovation Challenge (AIC) · Smart Manufacturing Track
### Perangkat: Raspberry Pi 4 Model B (4GB RAM) + USB Webcam + Sensor E18-D80NK

Folder ini merangkum seluruh aspek fisik, pengkabelan (*wiring*), proteksi sirkuit 3.3V, arsitektur terdistribusi edge-to-central, kalibrasi sensor di venue, dan strategi operasional hackathon luring 10 jam.

---

## 📁 Struktur Dokumen IoT Track

| Dokumen | Isi & Cakupan |
| :--- | :--- |
| **[`01_STRATEGI_OPERASIONAL_HACKATHON.md`](./01_STRATEGI_OPERASIONAL_HACKATHON.md)** | • Linimasa detail 10 jam lomba (09.00 - 20.00 WIB) di Fasilkom UI.<br>• Prosedur verifikasi LO Pra-Final (08.00 WIB) & checklist kepatuhan Rulebook.<br>• Strategi checkpoint commit Git wajib panitia.<br>• SOP Verifikasi Integritas Kode & Restart Server H-10 menit presentasi. |
| **[`02_IMPLEMENTASI_HARDWARE_DAN_EDGE.md`](./02_IMPLEMENTASI_HARDWARE_DAN_EDGE.md)** | • Topologi Distributed Edge-to-Central (RPi 4 $\rightarrow$ Central Laptop).<br>• Bill of Materials (BOM) & Skematik Sirkuit Elektronika lengkap.<br>• Proteksi Tegangan 3.3V GPIO & Resistor Pembatas Arus LED 220Ω.<br>• Kalibrasi Sensor IR E18-D80NK di conveyor meja kerja.<br>• Alur Closed-Loop Sub-Detik (< 150 ms) & Pengendalian Relay/Buzzer.<br>• Panduan deployment di venue & skenario cadangan *Mock Data Mode*. |

---

## ⚡ Ringkasan Cepat Konfigurasi Perangkat

* **Kredensial SSH:** `ssh dti@192.168.137.251` (hostname: `raspberrypi.local`, user/pass: `dti` / `dti`)
* **Hotspot Laptop:** SSID Laptop Windows (IP Gateway: `192.168.137.1`)
* **Endpoint Telemetri Central:** `POST http://192.168.137.1:8000/api/v1/inspections/run`
* **MJPEG Live Stream:** `http://192.168.137.251:8080/video_feed`
* **Pinout Header BCM:**
  - `GPIO 17`: Sensor IR E18-D80NK (Active LOW, internal pull-up)
  - `GPIO 18`: Active Buzzer 5V (PWM Alert)
  - `GPIO 27`: LED Hijau (Status PASS)
  - `GPIO 22`: LED Kuning (Status CONDITIONAL)
  - `GPIO 23`: LED Merah (Status FAIL)
  - `GPIO 25`: Modul Relay Conveyor Motor Cut-off
