import threading
from inputs import get_gamepad


class Gamepad:

    class Inputs:
        LEFT_X = "ABS_X"
        LEFT_Y = "ABS_Y"
        RIGHT_X = "ABS_RX"
        RIGHT_Y = "ABS_RY"
        LEFT_TRIGGER = "ABS_Z"
        RIGHT_TRIGGER = "ABS_RZ"
        A = "BTN_SOUTH"
        B = "BTN_EAST"
        X = "BTN_WEST"
        Y = "BTN_NORTH"
        LB = "BTN_TL"
        RB = "BTN_TR"


    def __init__(self):
        self._buttons = {}
        self._axes = {}
        self._lock = threading.Lock()
        self._running = True
        self.connected = False

        self._thread = threading.Thread(
            target=self._read_loop,
            daemon=True
        )
        self._thread.start()

    def _read_loop(self):
        while self._running:
            try:
                events = get_gamepad()  # Blocks until controller events occur
                self.connected = True

                for event in events:
                    with self._lock:
                        if event.ev_type == "Key":
                            self._buttons[event.code] = bool(event.state)

                        elif event.ev_type == "Absolute":
                            self._axes[event.code] = event.state

            except Exception as e:
                print(f"Gamepad error: {e}")
                self._buttons.clear()
                self._axes.clear()
                self.stop()

    def get_button(self, button: str) -> bool :
        code = getattr(Gamepad.Inputs, button.upper(), button)
        with self._lock:
            return self._buttons.get(code, False) or self._axes.get(code, 0.0)
    def get_joystick(self, axis: str) -> float :
            code = getattr(Gamepad.Inputs, axis.upper(), axis.upper())
            with self._lock:
                raw = self._axes.get(code, 0)
                return max(-1.0, min(1.0, raw / 27500.0))
    def get_trigger(self, axis: str) -> float :
        code = getattr(Gamepad.Inputs, axis.upper(), axis.upper())
        with self._lock:
            raw = self._axes.get(code, 0)
            return max(0.0, min(1.0, raw / 255.0))

    def stop(self):
        self._running = False
        if (self.connected):
            self._thread.join()
        self.connected = False