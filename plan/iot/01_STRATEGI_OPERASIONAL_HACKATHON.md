# NusaQC — Rencana Strategis & Operasional Hackathon 10 Jam Luring
### Lokasi: Gedung Baru Fasilkom UI · Waktu: Sabtu, 26 September 2026 (09.00 – 20.00 WIB)
### COMPFEST 18 AI Innovation Challenge (AIC) · Smart Manufacturing Track

Dokumen ini memuat jadwal detail per jam, pembagian tugas tim, strategi pemenuhan checkpoint git, dan mitigasi kepatuhan terhadap **Rulebook Babak Final AI Innovation Challenge (AIC) COMPFEST 18**.

---

## 1. Parameter Sistem & Konstanta Terpadu (Single Source of Truth)

Untuk mencegah inkonsistensi antar-anggota tim dan dokumen:

| Parameter | Nilai Resmi | Keterangan |
| :--- | :--- | :--- |
| **Edge Hardware** | Raspberry Pi 4 Model B (RAM 4GB) | BCM2711, 4 Core Cortex-A72 @ 1.5 GHz |
| **Edge OS & User** | Raspberry Pi OS Lite 64-bit / user: `dti` | Headless, Python 3.11+, ONNX Runtime CPU |
| **Edge Hostname & IP** | `raspberrypi.local` / `192.168.137.251` | Terhubung ke Mobile Hotspot Laptop |
| **Edge Video Stream** | `http://192.168.137.251:8080` | Endpoint: `/video_feed`, `/health`, `/snapshot` |
| **Central Host / Laptop** | `192.168.137.1` (atau `localhost:8000`) | Gateway Hotspot Laptop |
| **Central Ingestion API** | `POST http://192.168.137.1:8000/api/v1/inspections/run` | Multipart/form-data (`image`, `fish_family`, `lot_id`) |
| **Web Dashboard** | `http://localhost:3000` | Next.js Dasbor QC, Peta Slot, & Live Camera |
| **Baseline Git Commit** | `0cab218` (24 Agustus 2026, 14:19 WIB) | Titik acuan verifikasi LO Pra-Final |

---

## 2. Linimasa Operasional 10 Jam (09.00 – 20.00 WIB)

```text
08.00 ─────── 08.45 : Verifikasi Integritas Repository & Hardware oleh LO
08.45 ─────── 09.00 : Pembukaan Hackathon & Pengundian Urutan Presentasi WIZ.AI
                      ┌────────────────────────────────────────────────────────┐
09.00 ─────── 12.00 : │ FASE I: BASELINE (Build-the-Eval & Error Discovery)    │
                      └────────────────────────────────────────────────────────┘
                      [TAG CHECKPOINT: checkpoint-1-baseline @ 12.00 WIB]
                      ┌────────────────────────────────────────────────────────┐
12.00 ─────── 18.00 : │ FASE II: ITERATION (Dual Track: Eval & Product)       │
                      │ • 12.00 - 13.00: Istirahat / Sholat / Makan            │
                      │ • 15.00 - 16.00: Presentasi Progres ke Tim WIZ.AI      │
                      └────────────────────────────────────────────────────────┘
                      [TAG CHECKPOINT: checkpoint-2-iteration @ 15.00 WIB]
                      [TAG CHECKPOINT: checkpoint-3-integration @ 18.00 WIB]
                      ┌────────────────────────────────────────────────────────┐
18.00 ─────── 20.00 : │ FASE III: INTEGRATION & STABILIZATION                  │
                      └────────────────────────────────────────────────────────┘
20.00 WIB           : ⛔ CODE FREEZE MUTLAK [TAG: final-submission]
20.00 ─────── 20.30 : Pencatatan Hash SHA-1 & Pengundian Urutan Live Pitching
20.30 WIB           : 📤 Batas Akhir Unggah Evaluation Artifact (PDF)
23.59 WIB           : 📤 Batas Akhir Unggah Pitch Deck (PPTX/PDF)
```

---

## 3. Rincian Pengerjaan Per Fase

### A. Verifikasi Pra-Hackathon (08.00 – 08.45 WIB)
* **Kondisi Meja Tim:** Menata laptop kerja dan rig hardware IoT (webcam, rig conveyor/gantry, Raspberry Pi).
* **Prosedur Pemeriksaan LO:**
  1. Di laptop kerja, jalankan perintah git di hadapan LO:
     ```bash
     git log -1
     git status
     git log --all --since="2026-08-25 23:55:00"
     git reflog | head -20
     git stash list
     ```
  2. LO mencocokkan SHA-1 commit terakhir dengan catatan panitia (`0cab218`).
  3. Status working tree wajib `nothing to commit, working tree clean` di luar folder dependensi, build, dan `.env`.
* **Kondisi Raspberry Pi:**
  - Bersih total dari berkas proyek pra-final, tidak ada service `nusaqc-edge` yang berjalan.
  - Mematuhi Bagian 2.1 Poin 3 & Poin 4c: hanya OS bersih yang tersisa.

---

### B. Fase I: Baseline (09.00 – 12.00 WIB)
* **Fokus Utama:** Membangun infrastruktur evaluasi otomatis (*Build-the-eval*) dan memetakan titik kelemahan sistem awal.
* **Aktivitas AI Engineer:**
  1. Meng-generate skrip test suite otomatis di direktori `eval/` (`eval/slices.py`, `eval/metrics.py`, `eval/run_eval.py`).
  2. Menjalankan pengujian terhadap model MVP awal menggunakan data uji di `data/`.
  3. Menemukan dan mendokumentasikan kegagalan spesifik (*baseline error*):
     - False positive defek akibat pantulan cahaya kilau lendir (*specular glare*).
     - Penurunan akurasi pada kondisi pencahayaan redup (*low light*).
     - Latensi inferensi CPU Raspberry Pi yang mencapai 2.4 detik (bottleneck Model 2).
  4. Menyimpan output kuantitatif awal ke `eval/baseline_results.json`.
* **Aktivitas IoT & Webdev Engineer:**
  1. Pasca pukul 09.00 WIB, transfer berkas kode edge ke Raspberry Pi:
     ```bash
     scp -r edge run.sh dti@192.168.137.251:~/nusaqc/
     scp -r AI/model-1/ AI/model-2/ dti@192.168.137.251:~/nusaqc/AI/
     ```
  2. Hubungkan webcam USB, sensor IR E18-D80NK, 3-LED, buzzer, dan relay.
  3. Nyalakan server MJPEG edge (`python -m edge.main --cam 0 --mjpeg-port 8080 --central-url http://192.168.137.1:8000`).
  4. Verifikasi live video feed di browser (`http://192.168.137.251:8080/video_feed`).
* **Milestone 12.00 WIB:**
  ```bash
  git add eval/
  git commit -m "feat(eval): add automated test suite and record baseline evaluation metrics"
  git tag checkpoint-1-baseline
  git push origin main --tags
  ```

---

### C. Fase II: Iteration (12.00 – 18.00 WIB)
* **Fokus Utama:** Eksekusi perbaikan paralel pada **Evaluation Track** dan **Product Track**.
* **1. Evaluation Track (AI Model Optimization):**
  - Kuantisasi dinamis INT8 pada Model 2 (YOLOv8s) untuk memangkas latensi dari 2.440 ms menjadi ~1.000 ms.
  - Retraining dengan Specular Glare Augmentation dan tuning NMS untuk menekan false positive kilau lendir dari 18.4% ke <= 5.0%.
  - Retraining Model 1 dengan Cost-Sensitive Loss (W_Grade_C = 5.0) untuk menekan Critical Escape Rate Grade C -> Grade A menjadi 0.00%.
  - Jalankan ulang `eval/run_eval.py` dan catat peningkatan metrik ke `eval/iteration_results.json`.
* **2. Product Track (UX, IoT & Web Features):**
  - Stabilisasi telemetry data injection dari edge ke FastAPI backend (`/api/v1/inspections/run`).
  - Fitur ekspor sertifikat mutu resmi (PDF Certificate dengan QR Code hash SHA-256) pada modul dispatch.
  - Pengujian *Mock Hardware Mode* via `ENABLE_MOCK_HARDWARE=true` sebagai jaring pengaman live demo.
* **3. Sesi Presentasi WIZ.AI (15.00 – 16.00 WIB):**
  - Minimal 1 anggota tim memaparkan progres di hadapan tim WIZ.AI (Bahasa Inggris, langsung dari layar kerja):
    1. Progres pekerjaan sejak 09.00 WIB.
    2. Temuan utama kelemahan baseline dan tindakan perbaikan yang diambil.
    3. Rencana pengerjaan hingga code freeze.
* **Milestone 15.00 & 18.00 WIB:**
  ```bash
  git commit -m "feat(iteration): apply INT8 quantization, NMS tuning, and PDF certificate export"
  git tag checkpoint-2-iteration
  git push origin main --tags
  # Sebelum pukul 18.00:
  git commit -m "feat(integration): finalize dual-model edge pipeline and dashboard telemetry"
  git tag checkpoint-3-integration
  git push origin main --tags
  ```

---

### D. Fase III: Integration & Code Freeze (18.00 – 20.00 WIB)
* **Aturan Mutlak:** **Dilarang memodifikasi model AI atau logika inferensi setelah pukul 18.00 WIB.**
* **Aktivitas Tim:**
  1. *End-to-End Dry Run:* Uji coba satu siklus utuh: deteksi sensor $\rightarrow$ inferensi edge $\rightarrow$ aktuasi lampu/buzzer/relay $\rightarrow$ dasbor web $\rightarrow$ ekspor PDF.
  2. Pastikan footer/header dasbor menampilkan penanda versi commit SHA.
  3. Lakukan commit terakhir dan push tag sebelum 20.00 WIB:
     ```bash
     git tag final-submission
     git push origin main --tags
     ```
  4. Panitia mencatat SHA-1 hash commit terakhir pada pukul 20.00 WIB.

---

### E. Pasca Code Freeze (20.00 – 23.59 WIB)
1. **Pukul 20.30 WIB:** Batas akhir unggah **Evaluation Artifact (PDF)** ke portal `compfest.id`:
   - Rincian test suite (kasus uji, metrik, alasan pemilihan).
   - Temuan baseline error awal beserta bukti kuantitatif.
   - Tabel komparasi metrik sebelum vs sesudah iterasi.
   - Analisis kegagalan eksperimen dan rencana lanjutan.
2. **Pukul 23.59 WIB:** Batas akhir unggah **Pitch Deck (PPTX/PDF)** ke portal `compfest.id`:
   - Rasio slide 16:9, Bahasa Inggris, **tanpa nama/logo almamater kampus**.

---

## 4. Pembagian Peran Tim (Jobdesk 10 Jam)

| Peran | Anggota | Tanggung Jawab Utama |
| :--- | :--- | :--- |
| **AI Engineer** | Tim AI | Eksekusi `eval/`, analisis baseline error, kuantisasi INT8 Model 2, penyusunan naskah Evaluation Artifact PDF. |
| **IoT / Embedded Dev**| Tim Hardware | Perakitan gantry, koneksi pin GPIO, kalibrasi jarak sensor IR, deployment edge ke RPi, optimasi latensi C-code/Python. |
| **Fullstack / Lead** | Tim Web & Pitch | Backend FastAPI, WebSocket telemetry, ekspor PDF sertifikat mutu, presentasi WIZ.AI, finalisasi slide Pitch Deck. |

---

## 5. SOP Verifikasi Integritas Hari Ke-2 (Live Pitching)

Satu slot sebelum giliran tampil (30 menit sebelum jadwal pitching):
1. Tim membawa laptop demonstrasi ke **Ruang Verifikasi**.
2. Di hadapan LO, peserta menjalankan:
   ```bash
   git log -1
   git status
   ```
   LO mencocokkan hash SHA-1 commit dengan catatan panitia pada Code Freeze (20.00 WIB).
3. LO memastikan berkas konfigurasi (`.env`) hanya berisi variabel environment, bukan logika program baru.
4. **Live Restart pada H-10 menit:** Peserta mematikan server lokal (`docker compose down` atau Ctrl+C) lalu menyalakannya kembali persis dari folder yang diverifikasi.
