# NusaQC — Cetak Biru Implementasi IoT, Hardware, & Edge AI
### Target: Hackathon 10 Jam, Closed-Loop Sortasi, & Live Pitching COMPFEST 18 AIC
### Target Perangkat: Raspberry Pi 4 Model B (4GB RAM) + USB Webcam + Sensor E18-D80NK

Dokumen ini adalah **master reference** untuk arsitektur terdistribusi, perakitan fisik (*wiring & electronics*), proteksi tegangan 3.3V GPIO, kalibrasi sensor, optimasi latensi inferensi ARM Cortex-A72, skema deployment, dan panduan kontinjensi demo.

---

## 1. Topologi Sistem: Distributed Edge-to-Central

```text
       LINI PRODUKSI (PABRIK / MEJA DEMO)                 KANTOR QC / LAPTOP TIM
┌──────────────────────────────────────────────┐     
│  [EDGE NODE: RASPBERRY PI 4 (192.168.137.251)]│     
│  • Single Board Computer (Broadcom BCM2711)   │     
│  • USB Webcam UVC (Overhead Snapshot)        │     
│  • Sensor IR E18-D80NK (Active LOW GPIO 17)   │     
│  • 3-LED Traffic Light (PASS / COND / FAIL)   │────┐
│  • Active Buzzer 5V & Modul Relay Conveyor   │    │
│  • Dual ONNX Runtime CPU Inference Lokal      │    │
│  • Background Telemetry Worker (Non-blocking) │    │
│  • MJPEG Stream Server (:8080)                │    │
└──────────────────────────────────────────────┘    │ HTTP POST (multipart/form-data)
                                                    │ Snapshot JPEG + Grade + BBox
                                                    │ & MJPEG Live Stream (:8080)
                                                    ▼
                                       ┌─────────────────────────┐
                                       │  CENTRAL SERVER         │
                                       │  (Laptop Tim)           │
                                       │  • FastAPI Backend :8000│
                                       │  • SQLite Database      │
                                       │  • Next.js Frontend:3000│
                                       │  • WebSocket Broadcast  │
                                       │  • Live Edge Camera Cam │
                                       └─────────────────────────┘
```

---

## 2. Bill of Materials (BOM) & Komponen Fisik

| Komponen | Spesifikasi / Model | Peran & Fungsi dalam Sistem |
| :--- | :--- | :--- |
| **Edge Computer** | Raspberry Pi 4 Model B (4GB RAM) | Menjalankan dual inferensi ONNX, kontrol GPIO, dan MJPEG server. |
| **Kamera Inspeksi** | USB Webcam UVC (720p / 1080p Fixed Focus) | Akuisisi citra overhead conveyor ikan & live stream browser. |
| **Sensor Keberadaan Ikan**| E18-D80NK (Diffuse IR Photoelectric) | Deteksi instan saat ikan melintas di bawah kamera (< 2 ms respon). |
| **Indikator PASS** | LED 5mm Hijau + Resistor 220Ω seri | Menyala saat ikan Grade A dan bebas cacat fisik. |
| **Indikator CONDITIONAL**| LED 5mm Kuning + Resistor 220Ω seri | Menyala saat ikan Grade B / butuh inspeksi manual manusia. |
| **Indikator FAIL** | LED 5mm Merah + Resistor 220Ω seri | Menyala instan saat ikan Grade C (afkir) atau terdeteksi cacat berat. |
| **Peringatan Suara** | Active Buzzer 5V DC | Bunyi bip pada kasus CONDITIONAL dan alarm kontinyu pada FAIL. |
| **Aktuator Sortasi** | Modul Relay 1-Channel 5V (Optocoupler) | Memutus sirkuit motor conveyor atau mengaktifkan solenoid ejektor. |
| **Catu Daya Edge** | Adaptor USB-C 5V / 3A Official | Menjamin pasokan arus stabil tanpa *undervoltage throttle*. |

---

## 3. Peringatan Proteksi Tegangan 3.3V & Skema Pengkabelan (Wiring)

> ⚠️ **PERINGATAN KRITIS ELEKTRONIKA:**
> - Seluruh pin GPIO Raspberry Pi beroperasi pada level logika **3.3V MAKSIMAL** dan **TIDAK 5V-TOLERANT**.
> - Sensor E18-D80NK membutuhkan tegangan input 5V untuk pemancar inframerahnya.
> - Kabel output (Hitam) sensor adalah **NPN Open-Collector** (hanya menghubungkan ke GND saat mendeteksi objek).
> - **Solusi Aman:** Mengaktifkan **internal pull-up resistor 3.3V** pada Raspberry Pi (di kode: `gpiozero.Button(17, pull_up=True)`). Saat idle, pin tertahan di 3.3V; saat aktif, pin ditarik ke GND (0V). Sangat aman tanpa merusak SoC.

### Tabel Pemetaan Pinout Header Raspberry Pi 4 (BCM Pinout)

| Pin Fisik | Label BCM | Perangkat Terhubung | Keterangan Sambungan |
| :---: | :---: | :--- | :--- |
| **Pin 02** | **5V Power** | Sensor E18-D80NK & Relay VCC | Kabel Cokelat Sensor & VCC Relay (Rel 5V paralel) |
| **Pin 04** | **5V Power** | Kipas Pendingin Pi (FAN +) | Kabel Merah Cooling Fan |
| **Pin 06** | **GND** | Kipas Pendingin Pi (FAN -) | Kabel Hitam Cooling Fan |
| **Pin 09** | **GND** | Katoda Ketiga LED (GND Rail) | Kaki pendek (katoda) LED Hijau, Kuning, Merah |
| **Pin 11** | **GPIO 17** | **Sensor E18-D80NK (Signal)** | Kabel Hitam Sensor (Active LOW, Internal Pull-Up) |
| **Pin 12** | **GPIO 18** | **Active Buzzer (Signal)** | Pin sinyal positif (+) buzzer aktif (PWM) |
| **Pin 13** | **GPIO 27** | **LED Hijau (PASS)** | Anoda LED Hijau via Resistor 220Ω seri |
| **Pin 14** | **GND** | Sensor E18-D80NK (GND) | Kabel Biru Sensor |
| **Pin 15** | **GPIO 22** | **LED Kuning (CONDITIONAL)** | Anoda LED Kuning via Resistor 220Ω seri |
| **Pin 16** | **GPIO 23** | **LED Merah (FAIL)** | Anoda LED Merah via Resistor 220Ω seri |
| **Pin 20** | **GND** | Relay GND & Buzzer GND | Ground modul Relay & pin negatif (-) Buzzer |
| **Pin 22** | **GPIO 25** | **Relay Conveyor (IN)** | Pin sinyal pemicu relay (Active HIGH) |
| **USB 3.0** | **USB** | **USB Webcam** | Ditancapkan ke port biru USB 3.0 RPi |

### Perhitungan Resistor Pembatas Arus LED 220Ω
$$I = \frac{V_{GPIO} - V_f}{R} = \frac{3.3\text{V} - 2.0\text{V}}{220\Omega} \approx 5.9\text{ mA}$$
Arus ~5.9 mA sangat aman untuk pin GPIO RPi (maksimal 16 mA per pin) dan menjaga LED 5mm awet menyala terang.

---

## 4. Alur Kerja Closed-Loop Sub-Detik (< 150 ms)

```text
[ 1. SENSING ]
Ikan melintas di conveyor ──> Sensor E18-D80NK aktif ──> GPIO 17 mendeteksi sinyal LOW

[ 2. AKUISISI ]
Thread kamera membaca 1 snapshot frame langsung dari buffer memori webcam USB via OpenCV

[ 3. INFERENSI DUAL AI LOKAL (EDGE CPU) ]
├── Model 1: MobileNetV3-Small (Freshness) ──> Grade A / B / C (Latensi ~28 ms)
└── Model 2: YOLOv8s INT8 (Defect Detection) ──> Bounding Boxes Cacat (Latensi ~1.000 ms)

[ 4. DETERMINISTIC DECISION & INSTANT ACTUATION ]
Evaluasi aturan mutu SNI:
├── Status PASS (Grade A, tanpa defek):
│   └── GPIO 27 (LED Hijau) ON ──> Conveyor tetap jalan normal
├── Status CONDITIONAL (Grade B atau defek minor):
│   └── GPIO 22 (LED Kuning) ON ──> Buzzer bip singkat (periksa manual)
└── Status FAIL (Grade C atau defek mayor):
    └── GPIO 23 (LED Merah) ON ──> Buzzer alarm kontinyu ──> Relay GPIO 25 memutus motor conveyor

[ 5. BACKGROUND TELEMETRY DISPATCH (NON-BLOCKING) ]
Worker thread edge mengirimkan paket multipart/form-data ke Central Server:
POST http://192.168.137.1:8000/api/v1/inspections/run
- file: snapshot JPEG dengan visual bounding box teranotasi
- fish_family: "Scombridae"
- lot_id: "LOT-20260926-001"
Server menyimpan ke SQLite dan menyiarkan hasil secara instan via WebSocket ke dasbor.
```

---

## 5. Kalibrasi Sensor E18-D80NK di Meja Kerja

Di bagian belakang silinder sensor E18-D80NK terdapat sekrup potensiometer kecil dan lampu LED indikator merah bawaan sensor:
1. Hubungkan sensor ke daya 5V dan GND.
2. Tempatkan sampel ikan di atas meja conveyor tepat di bawah sensor.
3. Putar sekrup potensiometer menggunakan obeng minus kecil:
   - **Searah jarum jam (CW)**: Menambah jarak deteksi (hingga maks ~80 cm).
   - **Berlawanan jarum jam (CCW)**: Mengurangi jarak deteksi (hingga min ~3 cm).
4. **Target Kalibrasi Ideal**:
   - Meja kosong $\rightarrow$ LED indikator belakang sensor **PADAM**.
   - Ikan melintas $\rightarrow$ LED indikator belakang **MENYALA TEGAS**, dan langsung padam saat ikan lewat.

---

## 6. Analisis & Optimasi Latensi Edge AI (ARM Cortex-A72)

### A. Pengukuran Empiris Ground Truth di Raspberry Pi 4 (1.5 GHz)
* **Model 1 (MobileNetV3-Small FP32):**
  - 2 Thread: **28.3 ms**
  - 4 Thread: **33.0 ms**
  - *Kesimpulan:* Hanya menyumbang < 1.5% total latensi; tidak perlu optimasi.
* **Model 2 (YOLOv8s FP32 640x640):**
  - 2 Thread: **2,440.9 ms**
  - 4 Thread: **1,849.0 ms** (terpangkas ~600 ms / ~25% lebih cepat).
  - *Kesimpulan:* Bottleneck utama sistem; membutuhkan kuantisasi INT8.

### B. Tindakan Optimasi Nyata
1. **Thread Tuning (4 Core):** Mengatur `NUSAQC_ORT_THREADS=4` agar memanfaatkan 4 core CPU Cortex-A72.
2. **Kuantisasi Dinamis INT8 pada YOLOv8s:**
   - Memangkas ukuran model dari 43 MB ke **~11.5 MB**.
   - Memotong latensi di RPi 4 dari 1.849 ms menjadi **~950 – 1.100 ms (2.2x speedup)**.
3. **Pipeline Penuh Dual-AI (Tanpa Shortcut):**
   - Setiap sampel ikan selalu dieksekusi lengkap oleh Model 1 (Freshness ~28 ms) DAN Model 2 (Defect INT8 ~1.000 ms).
   - Menjamin 100% kelengkapan data inspeksi dan keabsahan log audit ekspor FDA/SNI, dengan total latensi agregat ~1.028 ms.
---

## 7. Prosedur Deployment di Hackathon (Pasca Pukul 09.00 WIB)

1. **Transfer Berkas Kode ke Raspberry Pi:**
   ```bash
   scp -r edge run.sh dti@192.168.137.251:~/nusaqc/
   scp -r AI/model-1/ AI/model-2/ dti@192.168.137.251:~/nusaqc/AI/
   ```
2. **Jalankan Aplikasi Edge di Pi:**
   ```bash
   ssh dti@192.168.137.251
   cd /home/dti/nusaqc
   python3 -m venv venv
   source venv/bin/activate
   pip install -r edge/requirements-edge.txt
   python -m edge.main --cam 0 --mjpeg-port 8080 --central-url http://192.168.137.1:8000
   ```
3. **Verifikasi Jalur Komunikasi:**
   - Buka browser: `http://192.168.137.251:8080/health` (Pastikan status `"ok"` dan `"camera_active": true`).
   - Buka frontend: `http://localhost:3000/inspection` (Pastikan widget *Live Edge Camera* berstatus hijau `LIVE`).

---

## 8. Panduan Koneksi Jaringan & SSH Troubleshooting

### Skema Jaringan di Venue Hackathon
```text
Internet ──> Router Venue ──> [WiFi] ──> Laptop Tim (Windows)
                                            │
                                     Mobile Hotspot
                                     (192.168.137.1)
                                            │
                                         [WiFi]
                                            │
                                     Raspberry Pi 4
                                     hostname: raspberrypi.local
                                     IP: 192.168.137.251 / user: dti
```

### Troubleshooting:
1. **RPi tidak muncul di `raspberrypi.local`**:
   - Cari IP RPi di panel Mobile Hotspot Windows (Settings $\rightarrow$ Network & Internet $\rightarrow$ Mobile Hotspot).
   - Akses langsung via IP: `ssh dti@192.168.137.251`.
2. **Ganti SSID Hotspot via nmcli**:
   - Jika berpindah jaringan, scan dan hubungkan:
     ```bash
     sudo nmcli dev wifi connect "<SSID_BARU>" password "<PASSWORD_BARU>"
     ```

---

## 9. Rencana Kontinjensi: Mock Data Mode

Sesuai aturan Rulebook Bagian 2.6 Poin 5:
> *"Produk berbasis hardware disarankan tetap memiliki mock data mode, yaitu mode di mana software dapat berjalan tanpa hardware, sebagai cadangan apabila terjadi kegagalan perangkat pada saat demonstrasi."*

- Jika terjadi kendala pada kabel GPIO, kamera, atau daya di venue:
  - Aktifkan toggle di web frontend: pilih mode **"Upload Berkas"** atau **"Kamera Laptop"**.
  - Backend FastAPI secara otomatis mengaktifkan pipeline inferensi lokal berbasis ONNX di laptop tim.
  - Demonstrasi fitur inti (deteksi kesegaran, visualisasi bounding box, penyimpanan lot, dan ekspor laporan PDF) dijamin tetap berjalan 100% tanpa hambatan di hadapan dewan juri.
