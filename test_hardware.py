#!/usr/bin/env python3
"""
NusaQC - Raspberry Pi Hardware Actuator & Sensor Test Script
Digunakan untuk menguji GPIO Pinout:
  - LED Hijau  : GPIO 27 (Pin 13)
  - LED Kuning : GPIO 22 (Pin 15)
  - LED Merah  : GPIO 23 (Pin 16)
  - Buzzer     : GPIO 18 (Pin 12)
  - Relay      : GPIO 25 (Pin 22)
  - Sensor IR  : GPIO 17 (Pin 11) - Active LOW

Dapat dijalankan langsung di Raspberry Pi:
  python3 test_hardware.py
"""

import time
import sys

# BCM Pin Definition (sesuai kontrak NusaQC)
PIN_IR_SENSOR = 17       # Pin 11 fisik (Active LOW)
PIN_BUZZER = 18          # Pin 12 fisik
PIN_LED_GREEN = 27       # Pin 13 fisik
PIN_LED_YELLOW = 22      # Pin 15 fisik
PIN_LED_RED = 23         # Pin 16 fisik
PIN_CONVEYOR_RELAY = 25  # Pin 22 fisik

def print_header():
    print("=" * 60)
    print("      🚀 NUSAQC RASPBERRY PI 4 HARDWARE TESTER")
    print("=" * 60)
    print(f"  • LED Hijau  (PASS)        : GPIO {PIN_LED_GREEN} (Pin 13)")
    print(f"  • LED Kuning (CONDITIONAL) : GPIO {PIN_LED_YELLOW} (Pin 15)")
    print(f"  • LED Merah  (FAIL)        : GPIO {PIN_LED_RED} (Pin 16)")
    print(f"  • Active Buzzer            : GPIO {PIN_BUZZER} (Pin 12)")
    print(f"  • Relay Konveyor           : GPIO {PIN_CONVEYOR_RELAY} (Pin 22)")
    print(f"  • Sensor IR E18-D80NK      : GPIO {PIN_IR_SENSOR} (Pin 11)")
    print("=" * 60)

# Coba inisialisasi driver GPIO (gpiozero atau RPi.GPIO)
BACKEND = None
try:
    from gpiozero import LED, Buzzer, OutputDevice, Button
    BACKEND = "gpiozero"
except ImportError:
    try:
        import RPi.GPIO as GPIO
        BACKEND = "RPi.GPIO"
    except ImportError:
        print("\n❌ Error: Library GPIO tidak ditemukan!")
        print("Silakan install salah satu library di RPi:")
        print("  sudo apt install python3-gpiozero python3-rpi.gpio")
        print("  atau:")
        print("  pip install rpi-lgpio gpiozero")
        sys.exit(1)

print(f"✅ Menggunakan backend GPIO: [{BACKEND}]")

class HardwareTester:
    def __init__(self):
        self.backend = BACKEND
        if self.backend == "gpiozero":
            self.led_green = LED(PIN_LED_GREEN)
            self.led_yellow = LED(PIN_LED_YELLOW)
            self.led_red = LED(PIN_LED_RED)
            self.buzzer = Buzzer(PIN_BUZZER)
            self.relay = OutputDevice(PIN_CONVEYOR_RELAY, initial_value=False)
            try:
                self.sensor = Button(PIN_IR_SENSOR, pull_up=True)
            except Exception:
                self.sensor = None
        else:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self.pins = [PIN_LED_GREEN, PIN_LED_YELLOW, PIN_LED_RED, PIN_BUZZER, PIN_CONVEYOR_RELAY]
            for p in self.pins:
                GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(PIN_IR_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def set_pin(self, pin_name, state: bool):
        if self.backend == "gpiozero":
            dev = getattr(self, pin_name, None)
            if dev:
                if state:
                    dev.on()
                else:
                    dev.off()
        else:
            pin_map = {
                "led_green": PIN_LED_GREEN,
                "led_yellow": PIN_LED_YELLOW,
                "led_red": PIN_LED_RED,
                "buzzer": PIN_BUZZER,
                "relay": PIN_CONVEYOR_RELAY
            }
            p = pin_map.get(pin_name)
            if p is not None:
                GPIO.output(p, GPIO.HIGH if state else GPIO.LOW)

    def read_sensor(self) -> bool:
        """Mengembalikan True jika ada objek (Active LOW)."""
        if self.backend == "gpiozero":
            return self.sensor.is_pressed if self.sensor else False
        else:
            return GPIO.input(PIN_IR_SENSOR) == GPIO.LOW

    def cleanup(self):
        print("\n🧹 Membersihkan state GPIO...")
        if self.backend == "gpiozero":
            self.led_green.off()
            self.led_yellow.off()
            self.led_red.off()
            self.buzzer.off()
            self.relay.off()
            self.led_green.close()
            self.led_yellow.close()
            self.led_red.close()
            self.buzzer.close()
            self.relay.close()
            if self.sensor:
                self.sensor.close()
        else:
            for p in self.pins:
                GPIO.output(p, GPIO.LOW)
            GPIO.cleanup()
        print("✅ Cleanup selesai. Semua actuator padam.")

def run_tests():
    print_header()
    tester = HardwareTester()

    try:
        # 1. Test LED Hijau
        print("\n[1/6] 🟢 Menguji LED HIJAU (GPIO 27)...")
        for i in range(2):
            tester.set_pin("led_green", True)
            print("   -> ON")
            time.sleep(0.5)
            tester.set_pin("led_green", False)
            print("   -> OFF")
            time.sleep(0.3)

        # 2. Test LED Kuning
        print("\n[2/6] 🟡 Menguji LED KUNING (GPIO 22)...")
        for i in range(2):
            tester.set_pin("led_yellow", True)
            print("   -> ON")
            time.sleep(0.5)
            tester.set_pin("led_yellow", False)
            print("   -> OFF")
            time.sleep(0.3)

        # 3. Test LED Merah
        print("\n[3/6] 🔴 Menguji LED MERAH (GPIO 23)...")
        for i in range(2):
            tester.set_pin("led_red", True)
            print("   -> ON")
            time.sleep(0.5)
            tester.set_pin("led_red", False)
            print("   -> OFF")
            time.sleep(0.3)

        # 4. Test All LEDs
        print("\n[4/6] 🚦 Menguji SEMUA LED BERSAMAAN (Traffic Light Check)...")
        tester.set_pin("led_green", True)
        tester.set_pin("led_yellow", True)
        tester.set_pin("led_red", True)
        print("   -> Semua LED menyala selama 1 detik...")
        time.sleep(1.0)
        tester.set_pin("led_green", False)
        tester.set_pin("led_yellow", False)
        tester.set_pin("led_red", False)
        print("   -> Semua LED padam.")

        # 5. Test Buzzer
        print("\n[5/6] 🔊 Menguji BUZZER (GPIO 18) - 3x Bip Singkat...")
        for i in range(3):
            print(f"   -> BIP {i+1}...")
            tester.set_pin("buzzer", True)
            time.sleep(0.15)
            tester.set_pin("buzzer", False)
            time.sleep(0.15)

        # 6. Test Relay
        print("\n[6/6] ⚡ Menguji RELAY KONVEYOR (GPIO 25)...")
        print("   -> Relay ON (Perhatikan bunyi 'klik' relay)...")
        tester.set_pin("relay", True)
        time.sleep(1.0)
        print("   -> Relay OFF...")
        tester.set_pin("relay", False)

        # Bonus: Tes Sensor IR
        print("\n" + "-" * 60)
        print("🔍 Membaca status Sensor IR E18-D80NK (GPIO 17) selama 5 detik...")
        print("   (Coba letakkan tangan/benda di depan sensor)")
        end_time = time.time() + 5.0
        last_state = None
        while time.time() < end_time:
            detected = tester.read_sensor()
            if detected != last_state:
                status_str = "🐟 OBJEK TERDETEKSI (LOW)" if detected else "⚪ KOSONG (HIGH)"
                print(f"   [{time.strftime('%H:%M:%S')}] {status_str}")
                last_state = detected
            time.sleep(0.1)

        print("\n" + "=" * 60)
        print("🎉 SEMUA PENGUJIAN HARDWARE SELESAI DENGAN SUKSES!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⚠️ Dihentikan oleh pengguna (Ctrl+C).")
    except Exception as e:
        print(f"\n❌ Terjadi kesalahan saat pengujian: {e}")
    finally:
        tester.cleanup()

if __name__ == "__main__":
    run_tests()
