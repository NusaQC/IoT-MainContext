import os
import sys
import time
import threading
from typing import Dict, Any, Optional

# Pin Definitions (BCM numbering)
PIN_IR_SENSOR = 17       # Active LOW, internal pull-up (Physical Pin 11)
PIN_BUZZER = 18          # Active Buzzer (Physical Pin 12)
PIN_LIGHT_GREEN = 27     # PASS indicator (Physical Pin 13)
PIN_LIGHT_YELLOW = 22    # CONDITIONAL indicator (Physical Pin 15)
PIN_LIGHT_RED = 23       # FAIL indicator (Physical Pin 16)
PIN_CONVEYOR_RELAY = 25  # Conveyor Relay, Active HIGH (Physical Pin 22)


class BaseGPIOController:
    """Abstract interface for GPIO controllers."""
    def trigger_signal(self, signal: str) -> Dict[str, Any]:
        raise NotImplementedError

    def is_fish_present(self) -> bool:
        raise NotImplementedError

    def get_status(self) -> Dict[str, Any]:
        raise NotImplementedError

    def cleanup(self) -> None:
        raise NotImplementedError


class HardwareGPIOController(BaseGPIOController):
    """
    Physical Raspberry Pi 4 GPIO Controller.
    Implements active low IR sensing, instant traffic light toggling,
    safety conveyor cut-off, and non-blocking buzzer pulses.
    """

    def __init__(self):
        import RPi.GPIO as GPIO
        self.GPIO = GPIO
        self.GPIO.setmode(self.GPIO.BCM)
        self.GPIO.setwarnings(False)

        # Configure outputs (relay, LEDs, buzzer)
        self._out_pins = [
            PIN_CONVEYOR_RELAY,
            PIN_LIGHT_GREEN,
            PIN_LIGHT_YELLOW,
            PIN_LIGHT_RED,
            PIN_BUZZER
        ]
        for pin in self._out_pins:
            self.GPIO.setup(pin, self.GPIO.OUT, initial=self.GPIO.LOW)

        # Configure input: E18-D80NK Active LOW with 3.3V internal pull-up
        self.GPIO.setup(PIN_IR_SENSOR, self.GPIO.IN, pull_up_down=self.GPIO.PUD_UP)

        # State tracking
        self.current_signal = "GREEN"
        self.conveyor_state = "ACTIVE"
        self.buzzer_state = "OFF"
        self._buzzer_thread: Optional[threading.Thread] = None
        self._buzzer_stop_event = threading.Event()

        # Initialize to standard PASS (Conveyor running, Green Light ON)
        self.trigger_signal("GREEN")
        print("⚡ [GPIO] Physical Raspberry Pi GPIO initialized (BCM mode).")

    def _beep_short(self, duration_s: float = 0.2):
        """Non-blocking single beep for CONDITIONAL alert."""
        def _worker():
            try:
                self.GPIO.output(PIN_BUZZER, self.GPIO.HIGH)
                time.sleep(duration_s)
                self.GPIO.output(PIN_BUZZER, self.GPIO.LOW)
            except Exception:
                pass
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def trigger_signal(self, signal: str) -> Dict[str, Any]:
        """
        Actuates GPIO hardware based on decision signal:
        - GREEN / PASS: Conveyor ON (HIGH), Green LED ON, Buzzer OFF
        - YELLOW / CONDITIONAL: Conveyor ON (HIGH), Yellow LED ON, Short Buzzer Beep
        - RED / FAIL: Conveyor STOP (LOW), Red LED ON, Continuous Buzzer Alarm
        """
        sig = signal.upper()
        if sig in ["PASS", "GREEN"]:
            self.GPIO.output(PIN_CONVEYOR_RELAY, self.GPIO.HIGH)  # Relay ON -> motor runs
            self.GPIO.output(PIN_LIGHT_GREEN, self.GPIO.HIGH)
            self.GPIO.output(PIN_LIGHT_YELLOW, self.GPIO.LOW)
            self.GPIO.output(PIN_LIGHT_RED, self.GPIO.LOW)
            self.GPIO.output(PIN_BUZZER, self.GPIO.LOW)
            self.current_signal = "GREEN"
            self.conveyor_state = "ACTIVE"
            self.buzzer_state = "OFF"

        elif sig in ["CONDITIONAL", "YELLOW"]:
            self.GPIO.output(PIN_CONVEYOR_RELAY, self.GPIO.HIGH)  # Relay ON -> motor runs
            self.GPIO.output(PIN_LIGHT_GREEN, self.GPIO.LOW)
            self.GPIO.output(PIN_LIGHT_YELLOW, self.GPIO.HIGH)
            self.GPIO.output(PIN_LIGHT_RED, self.GPIO.LOW)
            self._beep_short(0.2)
            self.current_signal = "YELLOW"
            self.conveyor_state = "ACTIVE"
            self.buzzer_state = "BEEP"

        elif sig in ["FAIL", "RED"]:
            self.GPIO.output(PIN_CONVEYOR_RELAY, self.GPIO.LOW)   # Relay OFF -> MOTOR CUT-OFF
            self.GPIO.output(PIN_LIGHT_GREEN, self.GPIO.LOW)
            self.GPIO.output(PIN_LIGHT_YELLOW, self.GPIO.LOW)
            self.GPIO.output(PIN_LIGHT_RED, self.GPIO.HIGH)
            self.GPIO.output(PIN_BUZZER, self.GPIO.HIGH)
            self.current_signal = "RED"
            self.conveyor_state = "STOPPED"
            self.buzzer_state = "ALARM"
        elif sig == "ALL_ON":
            self.GPIO.output(PIN_LIGHT_GREEN, self.GPIO.HIGH)
            self.GPIO.output(PIN_LIGHT_YELLOW, self.GPIO.HIGH)
            self.GPIO.output(PIN_LIGHT_RED, self.GPIO.HIGH)
            self.current_signal = "ALL_ON"
        elif sig == "ALL_OFF":
            self.GPIO.output(PIN_LIGHT_GREEN, self.GPIO.LOW)
            self.GPIO.output(PIN_LIGHT_YELLOW, self.GPIO.LOW)
            self.GPIO.output(PIN_LIGHT_RED, self.GPIO.LOW)
            self.GPIO.output(PIN_BUZZER, self.GPIO.LOW)
            self.current_signal = "ALL_OFF"
        return self.get_status()

    def is_fish_present(self) -> bool:
        """
        Reads E18-D80NK status on GPIO 17.
        Sensor is Active LOW: returns True when beam is reflected (object detected).
        """
        try:
            # Active LOW logic: 0 = Object present, 1 = Clear
            return self.GPIO.input(PIN_IR_SENSOR) == self.GPIO.LOW
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Returns current states of actuators and sensors."""
        sensor_active = self.is_fish_present()
        return {
            "mode": "PHYSICAL_GPIO",
            "signal": self.current_signal,
            "conveyor_relay": self.conveyor_state,
            "tower_light": self.current_signal,
            "buzzer": self.buzzer_state,
            "ir_sensor_detected": sensor_active,
            "pin_map": {
                "ir_sensor": PIN_IR_SENSOR,
                "buzzer": PIN_BUZZER,
                "led_green": PIN_LIGHT_GREEN,
                "led_yellow": PIN_LIGHT_YELLOW,
                "led_red": PIN_LIGHT_RED,
                "conveyor_relay": PIN_CONVEYOR_RELAY
            }
        }

    def cleanup(self) -> None:
        """Resets all pins to safe states and releases GPIO resources."""
        try:
            # Turn off all outputs
            for pin in self._out_pins:
                self.GPIO.output(pin, self.GPIO.LOW)
            self.GPIO.cleanup()
            print("⚡ [GPIO] Physical GPIO cleaned up successfully.")
        except Exception as e:
            print(f"⚠️ [GPIO] Cleanup warning: {e}")


class MockGPIOController(BaseGPIOController):
    """
    Simulation GPIO Controller with terminal visual feedback.
    Complies with Hackathon Mock Data Mode requirement.
    """

    def __init__(self):
        self.current_signal = "GREEN"
        self.conveyor_state = "ACTIVE"
        self.buzzer_state = "OFF"
        self._simulated_presence = False
        print("🎮 [GPIO MOCK] Mock GPIO Controller initialized (Simulation Mode).")
        self.trigger_signal("GREEN")

    def trigger_signal(self, signal: str) -> Dict[str, Any]:
        sig = signal.upper()
        if sig in ["PASS", "GREEN"]:
            self.current_signal = "GREEN"
            self.conveyor_state = "ACTIVE"
            self.buzzer_state = "OFF"
            color_code = "\033[92m"  # Green
        elif sig in ["CONDITIONAL", "YELLOW"]:
            self.current_signal = "YELLOW"
            self.conveyor_state = "ACTIVE"
            self.buzzer_state = "BEEP"
            color_code = "\033[93m"  # Yellow
        elif sig in ["FAIL", "RED"]:
            self.current_signal = "RED"
            self.conveyor_state = "STOPPED"
            self.buzzer_state = "ALARM"
            color_code = "\033[91m"  # Red
        elif sig == "ALL_ON":
            self.current_signal = "ALL_ON"
            color_code = "\033[96m"
        elif sig == "ALL_OFF":
            self.current_signal = "ALL_OFF"
            self.buzzer_state = "OFF"
            color_code = "\033[90m"
        else:
            self.current_signal = "UNKNOWN"
            color_code = "\033[0m"
        reset_code = "\033[0m"
        print(f"🎮 [GPIO MOCK] Actuation: {color_code}[{self.current_signal}]{reset_code} | "
              f"Conveyor: {self.conveyor_state} | Buzzer: {self.buzzer_state}")
        return self.get_status()

    def set_mock_presence(self, present: bool):
        self._simulated_presence = present

    def is_fish_present(self) -> bool:
        return self._simulated_presence

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": "MOCK_GPIO",
            "signal": self.current_signal,
            "conveyor_relay": self.conveyor_state,
            "tower_light": self.current_signal,
            "buzzer": self.buzzer_state,
            "ir_sensor_detected": self._simulated_presence,
            "pin_map": {
                "ir_sensor": PIN_IR_SENSOR,
                "buzzer": PIN_BUZZER,
                "led_green": PIN_LIGHT_GREEN,
                "led_yellow": PIN_LIGHT_YELLOW,
                "led_red": PIN_LIGHT_RED,
                "conveyor_relay": PIN_CONVEYOR_RELAY
            }
        }

    def cleanup(self) -> None:
        print("🎮 [GPIO MOCK] Cleaned up mock GPIO.")


def get_gpio_controller(force_mock: bool = False) -> BaseGPIOController:
    """
    Factory creating HardwareGPIOController if physical RPi.GPIO is available,
    otherwise automatically falling back to MockGPIOController.
    """
    if force_mock or os.environ.get("ENABLE_MOCK_HARDWARE", "").lower() in ["1", "true", "yes"]:
        return MockGPIOController()

    try:
        import RPi.GPIO
        return HardwareGPIOController()
    except Exception as e:
        print(f"ℹ️ [GPIO] Hardware GPIO unavailable ({e}). Using MockGPIOController.")
        return MockGPIOController()


# Backwards compatibility alias
GPIOController = get_gpio_controller
