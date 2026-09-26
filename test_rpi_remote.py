#!/usr/bin/env python3
"""
NusaQC - Remote Raspberry Pi Hardware Tester Runner
Jalankan script ini langsung dari Laptop:
  python test_rpi_remote.py

Script ini akan:
1. Konek ke Raspberry Pi (192.168.137.251) via SSH (user: dti / pass: dti)
2. Mentransfer test_hardware.py ke Raspberry Pi
3. Menjalankan test_hardware.py di Raspberry Pi secara interaktif
"""

import os
import sys
import paramiko

RPI_HOST = "192.168.137.251"
RPI_USER = "dti"
RPI_PASS = "dti"

def main():
    print("=" * 60)
    print("  🚀 NUSAQC REMOTE RASPBERRY PI TEST RUNNER")
    print(f"  Target: {RPI_USER}@{RPI_HOST}")
    print("=" * 60)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"\n[1/3] 🔌 Menghubungkan ke {RPI_HOST} via SSH...")
        ssh.connect(RPI_HOST, username=RPI_USER, password=RPI_PASS, timeout=10)
        print("   ✅ Terhubung!")

        # Pastikan test_hardware.py lokal ada
        local_script = os.path.join(os.path.dirname(__file__), "test_hardware.py")
        if not os.path.exists(local_script):
            local_script = "test_hardware.py"

        print("\n[2/3] 📤 Mengunggah test_hardware.py ke Raspberry Pi...")
        sftp = ssh.open_sftp()
        remote_path = "/home/dti/test_hardware.py"
        sftp.put(local_script, remote_path)
        sftp.close()
        print(f"   ✅ Script terunggah ke {remote_path}")

        print("\n[3/3] ⚡ Menjalankan pengujian hardware di Raspberry Pi...")
        print("-" * 60)
        
        # Eksekusi script di RPi
        stdin, stdout, stderr = ssh.exec_command(f"python3 {remote_path}", get_pty=True)

        for line in iter(stdout.readline, ""):
            print(line, end="")

        exit_status = stdout.channel.recv_exit_status()
        print("-" * 60)
        if exit_status == 0:
            print("\n🎉 Pengujian selesai dengan kode status 0 (SUKSES).")
        else:
            print(f"\n⚠️ Pengujian selesai dengan kode error {exit_status}.")

    except Exception as e:
        print(f"\n❌ Gagal terhubung atau mengeksekusi: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
