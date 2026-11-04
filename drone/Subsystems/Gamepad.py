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

    THRESHOLD = 0.2 #Deadzone percent threshold for joystick inputs
    def __init__(self):
        self._buttons = {}
        self._axes = {}
        self._lock = threading.Lock()
        self._running = True
        self.connected = False
        self.THRESHOLD = 0.2

        self._thread = threading.Thread(
            target=self._read_loop,
            daemon=True
        )
        self._thread.start()

    def _read_loop(self):
        while self._running:
            try:
                events = get_gamepad()  # Blocks until controller events occur
                # A blocked read may return after stop(); do not restore
                # commands on a stopped reader. Apply each batch atomically.
                with self._lock:
                    if not self._running:
                        break
                    self.connected = True
                    for event in events:
                        if event.ev_type == "Key":
                            self._buttons[event.code] = bool(event.state)
                        elif event.ev_type == "Absolute":
                            self._axes[event.code] = event.state

            except Exception as e:
                print(f"Gamepad error: {e}")
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
        # inputs.get_gamepad() can block until the device changes. Never wait
        # indefinitely, and never join this thread from its own error handler.
        with self._lock:
            self._running = False
            self.connected = False
            self._buttons.clear()
            self._axes.clear()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=0.2)

    @staticmethod
    def throttle_with_deadzone(value: float, threshold: float) -> float:
        """Continuous unipolar throttle: deadzone edge is zero, full stick is one.

        A hard cutoff otherwise makes the minimum nonzero thrust equal to
        the deadzone fraction, which can exceed hover on powerful quads.
        """
        if not 0 <= threshold < 1:
            raise ValueError("Throttle deadzone must be in [0, 1)")
        return max(0.0, min(1.0, (value - threshold) / (1.0 - threshold)))

    def rate_control_inputs(self) -> tuple[float, float, float, float]:
        throttle = self.get_joystick(Gamepad.Inputs.LEFT_Y)
        roll_vel     = self.get_joystick(Gamepad.Inputs.LEFT_X)
        pitch_vel    = self.get_joystick(Gamepad.Inputs.RIGHT_Y) * -1
        yaw_vel      = self.get_joystick(Gamepad.Inputs.RIGHT_X)

        # Apply deadzone threshold
        throttle = self.throttle_with_deadzone(throttle, self.THRESHOLD)
        if abs(roll_vel) < self.THRESHOLD:
            roll_vel = 0.0
        if abs(pitch_vel) < self.THRESHOLD:
            pitch_vel = 0.0
        if abs(yaw_vel) < self.THRESHOLD:
            yaw_vel = 0.0
        return throttle, roll_vel, pitch_vel, yaw_vel

    def attitude_control_inputs(self) -> tuple[float, float, float, float]:
        throttle = self.get_joystick(Gamepad.Inputs.LEFT_Y)
        roll     = self.get_joystick(Gamepad.Inputs.LEFT_X)
        pitch    = self.get_joystick(Gamepad.Inputs.RIGHT_Y) * -1
        yaw_vel      = self.get_joystick(Gamepad.Inputs.RIGHT_X)

        # Apply deadzone threshold
        throttle = self.throttle_with_deadzone(throttle, self.THRESHOLD)
        if abs(roll) < self.THRESHOLD:
            roll = 0.0
        if abs(pitch) < self.THRESHOLD:
            pitch = 0.0
        if abs(yaw_vel) < self.THRESHOLD:
            yaw_vel = 0.0
        return throttle, roll, pitch, yaw_vel

    def velocity_control_inputs(self) -> tuple[float, float, float, float]:
        vertical_vel = self.get_joystick(Gamepad.Inputs.LEFT_Y)
        horizontal_vel = self.get_joystick(Gamepad.Inputs.LEFT_X)
        forward_vel   = self.get_joystick(Gamepad.Inputs.RIGHT_Y) * -1
        yaw_vel      = self.get_joystick(Gamepad.Inputs.RIGHT_X)

        # Apply deadzone threshold
        if abs(vertical_vel) < self.THRESHOLD:
            vertical_vel = 0.0
        if abs(horizontal_vel) < self.THRESHOLD:
            horizontal_vel = 0.0
        if abs(forward_vel) < self.THRESHOLD:
            forward_vel = 0.0
        if abs(yaw_vel) < self.THRESHOLD:
            yaw_vel = 0.0
        return vertical_vel, horizontal_vel, forward_vel, yaw_vel