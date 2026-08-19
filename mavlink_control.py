import threading
import time
from typing import List, Optional

try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import common as mavlink
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "pymavlink is required for MAVLink control. Install it with `pip install pymavlink`."
    ) from exc

NEUTRAL_CHANNELS = (1500, 1500, 1500, 1500, 0, 0, 0, 0)
LAND_CHANNELS = (1500, 1500, 1300, 1500, 0, 0, 0, 0)

GESTURE_TO_RC: dict[int, tuple[int, int, int, int]] = {
    0: (1500, 1525, 1500, 1500),  # Forward => slow pitch forward
    1: (1500, 1475, 1500, 1500),  # Back => slow pitch backward
    2: (1500, 1500, 1525, 1500),  # Up => slow throttle increase
    3: (1500, 1500, 1475, 1500),  # Down => slow throttle decrease
    4: (1475, 1500, 1500, 1500),  # Left => slow roll left
    5: (1525, 1500, 1500, 1500),  # Right => slow roll right
    6: (1500, 1500, 1500, 1500),  # Stop => neutral controls
}


class MavlinkGestureController:
    """Simple MAVLink controller for Pixhawk via a serial link."""

    def __init__(
        self,
        connection_string: str,
        baud: int = 57600,
        target_system: int = 1,
        target_component: int = 1,
        source_system: int = 255,
        heartbeat_timeout: float = 5.0,
    ):
        self.connection_string = connection_string
        self.baud = baud
        self.target_system = target_system
        self.target_component = target_component
        self.source_system = source_system
        self._last_mode = "UNKNOWN"
        self._armed = False
        self._last_heartbeat: Optional[object] = None
        if "0.0.0.0" in connection_string.lower():
            print("[MAVLINK WARNING] Using '0.0.0.0' is usually wrong for sending commands to DroneBridge; it listens on all interfaces instead of targeting a specific bridge endpoint.")
            print("[MAVLINK WARNING] If Mission Planner / ArduPilot is already using 14550, you cannot bind to the same UDP port on the same machine.")
            print("[MAVLINK WARNING] Try: udp:127.0.0.1:14550, udp:<bridge-ip>:14550, or a different local listener such as udpin:0.0.0.0:14551.")
        self.master = mavutil.mavlink_connection(
            connection_string,
            baud=baud,
            source_system=source_system,
        )

        print(f"MAVLink: opening connection to {connection_string} at {baud} baud...")
        heartbeat = self._wait_for_heartbeat(heartbeat_timeout)
        if heartbeat is None:
            raise RuntimeError("No MAVLink heartbeat received.")

        self.target_system = getattr(heartbeat, "sysid", getattr(heartbeat, "srcSystem", self.target_system))
        self.target_component = getattr(heartbeat, "compid", getattr(heartbeat, "srcComponent", self.target_component))
        print(
            f"MAVLink: heartbeat received from system {self.target_system} component {self.target_component}.")
        self._last_command_id: Optional[int] = None
        self.refresh_state()

    def _wait_for_heartbeat(self, timeout: float):
        return self.master.wait_heartbeat(timeout=timeout)

    def _resolve_mode_name(self, msg: object) -> str:
        try:
            custom_mode = getattr(msg, "custom_mode", None)
            if custom_mode is not None:
                try:
                    custom_mode_int = int(custom_mode)
                except (TypeError, ValueError):
                    custom_mode_int = None
                if custom_mode_int is not None:
                    mapping = self.master.mode_mapping()
                    if mapping:
                        for name, value in mapping.items():
                            if int(value) == custom_mode_int:
                                return str(name).upper()
                    return f"Mode({custom_mode_int})"
        except Exception:
            pass

        try:
            return str(mavutil.mode_string_v10(msg)).upper()
        except Exception:
            return "UNKNOWN"

    def refresh_state(self) -> tuple[str, bool]:
        """Poll for the latest HEARTBEAT and cache mode + arm state."""
        if not getattr(self, "master", None):
            return self._last_mode, self._armed

        deadline = time.time() + 0.25
        while time.time() < deadline:
            msg = self.master.recv_match(blocking=False, timeout=0.05)
            if msg is None:
                continue
            if msg.get_type() == "HEARTBEAT":
                self._last_heartbeat = msg
                self._last_mode = self._resolve_mode_name(msg)
                self._armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                break
        return self._last_mode, self._armed

    def get_mode(self) -> str:
        self.refresh_state()
        return self._last_mode

    def is_armed(self) -> bool:
        self.refresh_state()
        return self._armed

    def is_manual(self) -> bool:
        return self.get_mode().upper() == "MANUAL"

    def close(self) -> None:
        timer = getattr(self, "_command_timer", None)
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass
            self._command_timer = None
        if self.master:
            try:
                self.send_neutral()
            except Exception:
                pass
            self.master.close()

    def _to_rc_channels(self, values: tuple[int, int, int, int]) -> List[int]:
        return [values[0], values[1], values[2], values[3], 0, 0, 0, 0]

    def send_rc_override(self, channel_values: tuple[int, int, int, int]) -> None:
        channels = self._to_rc_channels(channel_values)
        self.master.mav.rc_channels_override_send(
            self.target_system,
            self.target_component,
            channels[0],
            channels[1],
            channels[2],
            channels[3],
            channels[4],
            channels[5],
            channels[6],
            channels[7],
        )

    def send_neutral(self) -> None:
        self.master.mav.rc_channels_override_send(
            self.target_system,
            self.target_component,
            *NEUTRAL_CHANNELS,
        )

    def send_land_command(self) -> None:
        self.master.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavlink.MAV_CMD_NAV_LAND,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def send_rtl_command(self) -> None:
        try:
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )
        except Exception as exc:
            print(f"[MAVLINK] RTL command failed: {exc}")
            raise

    def trigger_rtl(self, reason: str = "Safety watchdog") -> None:
        print(f"[MAVLINK SAFETY] {reason}. Enabling RTL.")
        try:
            self.send_neutral()
        except Exception:
            pass
        try:
            self.set_mode("RTL")
        except Exception:
            pass
        try:
            self.send_rtl_command()
        except Exception:
            pass

    def ensure_safe_state(self) -> bool:
        try:
            mode = self.get_mode().upper()
            armed = self.is_armed()
        except Exception:
            return False

        if not armed:
            print("[MAVLINK SAFETY] Vehicle is not armed; safety abort triggered.")
            self.trigger_rtl("Vehicle disarmed unexpectedly")
            return False

        if mode in {"MANUAL", "STABILIZE", "ACRO", "RATTITUDE", "ALT_HOLD", "POSHOLD", "LOITER"}:
            if mode not in {"GUIDED", "RTL"}:
                print(f"[MAVLINK SAFETY] Unsafe control mode detected: {mode}. Engaging RTL.")
                self.trigger_rtl(f"Unsafe mode {mode}")
                return False

        return True

    def wait_for_gps_lock(self, timeout: float = 20.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            try:
                msg = self.master.recv_match(blocking=False, timeout=0.2)
                if msg is not None and msg.get_type() == "GPS_RAW_INT":
                    fix_type = int(getattr(msg, "fix_type", 0))
                    if fix_type >= 3:
                        return True
            except Exception:
                pass
            time.sleep(0.1)
        print("[MAVLINK SAFETY] GPS fix not ready before takeoff.")
        return False

    def safe_startup(self, altitude_m: float, timeout: float = 60.0) -> tuple[bool, str]:
        try:
            self.refresh_state()
            current_mode = self.get_mode().upper()
            if current_mode in {"RTL", "LAND"}:
                return False, "Vehicle is already in RTL/Land; startup aborted."

            if not self.wait_for_gps_lock(timeout=20.0):
                self.trigger_rtl("No GPS fix before takeoff")
                return False, "GPS lock not ready; startup aborted."

            if not self.set_mode("GUIDED"):
                self.trigger_rtl("GUIDED mode set failed")
                return False, "Unable to set GUIDED mode."

            deadline = time.time() + timeout
            while time.time() < deadline:
                if self.get_mode().upper() == "GUIDED":
                    break
                time.sleep(0.2)
            else:
                self.trigger_rtl("GUIDED mode not confirmed")
                return False, "GUIDED mode not confirmed within timeout."

            if not self.is_armed():
                self.arm(force=False)
                deadline = time.time() + 15.0
                while time.time() < deadline:
                    if self.is_armed():
                        break
                    time.sleep(0.2)
                if not self.is_armed():
                    self.trigger_rtl("Arm confirmation failed")
                    return False, "Arm confirmation failed."

            self.takeoff(altitude_m)
            hover_ok, hover_altitude = self.wait_for_stable_hover(
                target_altitude=altitude_m,
                tolerance=0.35,
                required_reads=3,
                timeout=60.0,
            )
            if not hover_ok:
                self.trigger_rtl("Hover not stable at target altitude")
                return False, f"Hover not stable within timeout. Last alt={hover_altitude:.2f}m"

            return True, f"Stable hover at {hover_altitude:.2f}m"
        except Exception as exc:
            print(f"[MAVLINK SAFETY] Startup failed: {exc}")
            try:
                self.trigger_rtl("Startup failure")
            except Exception:
                pass
            return False, f"Startup failed: {exc}"

    def update_gesture(self, gesture_id: int, duration: float = 0.0) -> None:
        if getattr(self, "_command_timer", None) is not None:
            try:
                self._command_timer.cancel()
            except Exception:
                pass
            self._command_timer = None

        if gesture_id == 7:
            self.send_land_command()
            self._last_command_id = gesture_id
            return

        if gesture_id in GESTURE_TO_RC:
            self.send_rc_override(GESTURE_TO_RC[gesture_id])
            self._last_command_id = gesture_id
            if duration > 0:
                self._command_timer = threading.Timer(duration, self.send_neutral)
                self._command_timer.daemon = True
                self._command_timer.start()
            return

        self.send_neutral()
        self._last_command_id = None

    def set_mode(self, mode_name: str) -> bool:
        try:
            name = mode_name.upper()
            mapping = self.master.mode_mapping()
            if mapping and name in {k.upper(): v for k, v in mapping.items()}:
                mode_id = {k.upper(): v for k, v in mapping.items()}[name]
                self.master.mav.set_mode_send(
                    self.target_system,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    mode_id,
                )
                time.sleep(0.2)
                self.refresh_state()
                return True

            self.master.set_mode_apm(mode_name)
            time.sleep(0.2)
            self.refresh_state()
            return True
        except Exception as exc:
            print(f"[MAVLINK] set_mode('{mode_name}') failed: {exc}")
            return False

    def arm(self, force: bool = False) -> None:
        """Arm the vehicle normally. Use force=True only for special indoor bench testing."""
        try:
            force_value = 21196.0 if force else 0.0
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,
                force_value,
                0,
                0,
                0,
                0,
                0,
            )
            time.sleep(0.2)
            self.refresh_state()
            return
        except Exception as exc:
            print(f"[MAVLINK] arm(force={force}) failed: {exc}")
            raise

    def ensure_manual_and_armed(self, timeout: float = 10.0) -> bool:
        """Retry normal armed transition. This is kept for compatibility but not used in guided-hover startup."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                self.set_mode("MANUAL")
            except Exception:
                pass
            try:
                self.arm(force=False)
            except Exception:
                pass
            self.refresh_state()
            if self.get_mode().upper() == "MANUAL" and self.is_armed():
                return True
            time.sleep(0.5)
        return self.get_mode().upper() == "MANUAL" and self.is_armed()

    def takeoff(self, altitude_m: float) -> None:
        self.master.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            altitude_m,
        )

    def wait_for_stable_hover(self, target_altitude: float, tolerance: float = 0.35, required_reads: int = 3, timeout: float = 60.0) -> tuple[bool, float]:
        start = time.time()
        stable_count = 0
        last_altitude = 0.0

        while time.time() - start < timeout:
            msg = self.master.recv_match(blocking=False, timeout=0.25)
            if msg is not None and msg.get_type() == "GLOBAL_POSITION_INT":
                last_altitude = msg.relative_alt / 1000.0

            if last_altitude >= target_altitude - tolerance:
                stable_count += 1
                if stable_count >= required_reads:
                    self.send_neutral()
                    return True, last_altitude
            else:
                stable_count = 0

            time.sleep(0.2)

        return False, last_altitude

    def get_last_command(self) -> Optional[int]:
        return self._last_command_id
