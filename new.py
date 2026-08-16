import json
import os
import time
import threading

from vosk import KaldiRecognizer, Model  # type: ignore
import speech_recognition as sr
from pymavlink import mavutil

recognizer = sr.Recognizer()
microphone = sr.Microphone()

# ==============================================================================
# 0. LOAD OFFLINE VOICE MODEL FIRST
# ==============================================================================
MODEL_PATH = "model"
if not os.path.exists(MODEL_PATH):
    raise Exception(
        f"Vosk model folder not found at '{MODEL_PATH}'. Please download and"
        " extract it."
    )

print("Loading offline speech model...")
vosk_model = Model(MODEL_PATH)
print("Speech model loaded.")

# ==============================================================================
# 1. VEHICLE CONNECTION
# ==============================================================================
print("Connecting to vehicle...")
vehicle = mavutil.mavlink_connection('udp:0.0.0.0:14550')
vehicle.wait_heartbeat()
target_system = vehicle.target_system
target_component = vehicle.target_component

print(f"Connected to System {target_system}! Flight Mode: {vehicle.flightmode}")

# ------------------------------------------------------------------------------
# Single-reader architecture: ONLY _reader_loop calls recv_match(). All other
# code reads from `state`. Do NOT add any other thread that calls recv_match()
# directly (that was the bug in the previous run) — route new message types
# through this loop and the state dict instead.
# ------------------------------------------------------------------------------
mav_lock = threading.Lock()
state_lock = threading.Lock()

state = {
    "heartbeat": None,          "heartbeat_ts": 0,
    "gps_raw": None,            "gps_raw_ts": 0,
    "global_position": None,    "global_position_ts": 0,
    "params": {},
}

STALE_THRESHOLD = 2.0  # seconds — data older than this is treated as untrustworthy

_hb_success_count = 0  # FIX: was referenced via `global` but never initialized


def _reader_loop():
    while True:
        try:
            msg = vehicle.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue
            mtype = msg.get_type()
            now = time.time()

            if mtype == 'STATUSTEXT':
                print(f"[FC MSG] {msg.text}")

            elif mtype == 'HEARTBEAT':
                with state_lock:
                    state["heartbeat"] = msg
                    state["heartbeat_ts"] = now

            elif mtype == 'GPS_RAW_INT':
                with state_lock:
                    state["gps_raw"] = msg
                    state["gps_raw_ts"] = now

            elif mtype == 'GLOBAL_POSITION_INT':
                with state_lock:
                    state["global_position"] = msg
                    state["global_position_ts"] = now

            elif mtype == 'PARAM_VALUE':
                with state_lock:
                    state["params"][msg.param_id] = msg.param_value

        except Exception as e:
            print(f"[READER ERROR] {type(e).__name__}: {e} — reader continuing")
            time.sleep(0.1)


threading.Thread(target=_reader_loop, daemon=True).start()


def _send_heartbeat_loop():
    global _hb_success_count
    while True:
        try:
            with mav_lock:
                vehicle.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0
                )
            _hb_success_count += 1
            if _hb_success_count % 10 == 0:
                print(f"[HEARTBEAT OK] {_hb_success_count} sent so far")
        except Exception as e:
            print(f"[HEARTBEAT ERROR] {type(e).__name__}: {e} — will retry in 1s")
        time.sleep(1)


threading.Thread(target=_send_heartbeat_loop, daemon=True).start()
print("Reader thread and heartbeat stream started.")


# ------------------------------------------------------------------------------
# Helper functions — all read from state, none call recv_match() directly
# ------------------------------------------------------------------------------
def get_mode():
    with state_lock:
        msg, ts = state["heartbeat"], state["heartbeat_ts"]
    if msg and (time.time() - ts) > STALE_THRESHOLD:
        print("[WARNING] HEARTBEAT data is stale — mode reading may be outdated!")
    return mavutil.mode_string_v10(msg) if msg else vehicle.flightmode


def is_armed():
    with state_lock:
        msg, ts = state["heartbeat"], state["heartbeat_ts"]
    if msg and (time.time() - ts) > STALE_THRESHOLD:
        print("[WARNING] HEARTBEAT data is stale — armed status may be outdated!")
    return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) if msg else False


def is_armable():
    with state_lock:
        gps = state["gps_raw"]
    return bool(gps and gps.fix_type >= 3)


def set_mode(mode_name):
    if mode_name not in vehicle.mode_mapping():
        print(f"Mode {mode_name} not available!")
        return
    mode_id = vehicle.mode_mapping()[mode_name]
    with mav_lock:
        vehicle.mav.set_mode_send(
            target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )


def set_mode_confirmed(mode_name, timeout=5):
    start = time.time()
    set_mode(mode_name)

    while True:
        current_mode = get_mode()
        if current_mode == mode_name:
            print(f"Mode confirmed: {mode_name}")
            return True

        if time.time() - start > timeout:
            print(f"WARNING: Mode change to {mode_name} not confirmed within {timeout}s")
            return False

        set_mode(mode_name)
        time.sleep(0.5)


def arm():
    with mav_lock:
        vehicle.mav.command_long_send(
            target_system, target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )


def set_param(name, value):
    with mav_lock:
        vehicle.mav.param_set_send(
            target_system, target_component,
            name.encode('utf-8'),
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )


def get_param(name, timeout=3):
    with mav_lock:
        vehicle.mav.param_request_read_send(
            target_system, target_component,
            name.encode('utf-8'), -1
        )
    start = time.time()
    while time.time() - start < timeout:
        with state_lock:
            if name in state["params"]:
                return state["params"][name]
        time.sleep(0.05)
    return None


def get_global_position(timeout=2):
    start = time.time()
    while time.time() - start < timeout:
        with state_lock:
            pos, ts = state["global_position"], state["global_position_ts"]
        if pos is not None and (time.time() - ts) < STALE_THRESHOLD:
            return pos
        time.sleep(0.05)
    print("[WARNING] GLOBAL_POSITION_INT is stale or missing — returning last known value.")
    with state_lock:
        return state["global_position"]


# Startup diagnostics
print("FS_GCS_ENABLE  =", get_param("FS_GCS_ENABLE"))
print("FS_GCS_TIMEOUT =", get_param("FS_GCS_TIMEOUT"))
print("FS_EKF_THRESH  =", get_param("FS_EKF_THRESH"))
print("FENCE_RADIUS   =", get_param("FENCE_RADIUS"))


# ------------------------------------------------------------------------------
# Flight Routines
# ------------------------------------------------------------------------------
def arm_and_takeoff(target_altitude):
    print("\nBASIC PREARM CHECKINGS START")

    while not is_armable():
        print(" Waiting for 3D GPS lock...")
        time.sleep(1)

    print("Setting mode to GUIDED...")
    set_mode("GUIDED")
    time.sleep(2)

    print("Arming motors...")
    arm()

    while not is_armed():
        print(" Waiting for motors to arm...")
        arm()  # Resend arm command if missed
        time.sleep(1)

    print("Successfully armed!")

    print("FINALLY TAKING OFF YOUR AEROSIGN DRONE")
    with mav_lock:  # FIX: this send bypassed the lock before
        vehicle.mav.command_long_send(
            target_system, target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, target_altitude
        )

    # Require several consecutive good readings, not just one, before declaring
    # "reached altitude" — a single wind/baro noise spike in relative_alt would
    # otherwise trigger a premature exit while the drone is still low.
    stable_count = 0
    REQUIRED_STABLE_READINGS = 3  # ~3 seconds of sustained altitude

    while True:
        pos = get_global_position(timeout=1)
        if pos:
            alt = pos.relative_alt / 1000.0  # mm -> m
            print(f" Current Altitude: {alt:.2f} m")
            if alt >= target_altitude * 0.95:
                stable_count += 1
                if stable_count >= REQUIRED_STABLE_READINGS:
                    print("Reached target altitude and holding steady!")
                    break
            else:
                stable_count = 0  # any dip resets the streak
        time.sleep(1)


def send_ned_velocity(velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle based on velocity relative to body heading:
    vx : Forward (+) / Backward (-)
    vy : Right (+)   / Left (-)
    vz : Down (+)    / Up (-)
    """
    msg = vehicle.mav.set_position_target_local_ned_encode(
        0,                                   # time_boot_ms
        target_system, target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # relative to drone nose
        0b0000101111000111,                  # type_mask (only speeds & yaw rate)
        0, 0, 0,                             # positions (ignored)
        velocity_x, velocity_y, velocity_z,  # velocities in m/s
        0, 0, 0,                             # accelerations (ignored)
        0, 0                                 # yaw, yaw_rate
    )

    for _ in range(int(5 * duration)):
        with mav_lock:  # FIX: this send bypassed the lock before
            vehicle.mav.send(msg)
        time.sleep(0.2)


def condition_yaw(heading, relative=True, direction=1):
    """Rotates drone heading. direction: 1 = CW, -1 = CCW"""
    with mav_lock:  # FIX: this send bypassed the lock before
        vehicle.mav.command_long_send(
            target_system, target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            heading,               # target angle in degrees
            0,                     # speed deg/s (0 = default)
            direction,             # 1 = CW, -1 = CCW
            1 if relative else 0,  # relative offset
            0, 0, 0
        )


# ------------------------------------------------------------------------------
# Voice command handling
# ------------------------------------------------------------------------------
def evaluate_voice_command(text):
    text = text.lower().strip()

    # Guard: directional flight commands only run in GUIDED mode
    if text in ["forward", "backward", "left", "right", "up", "down"]:
        current_mode = get_mode()
        if current_mode != "GUIDED":
            print(f"Command '{text}' ignored — vehicle not in GUIDED mode (current: {current_mode})")
            return

    if text == "go":
        print("Executing: Pitch Forward")
        send_ned_velocity(1.0, 0, 0, 2)

    elif text == "back":
        print("Executing: Pitch Backward")
        send_ned_velocity(-1.0, 0, 0, 2)

    elif text == "right":
        print("Executing: Strafe Left")
        send_ned_velocity(0, -1.0, 0, 2)

    elif text == "left":
        print("Executing: Strafe Right")
        send_ned_velocity(0, 1.0, 0, 2)

    elif text == "e":
        print("Executing: Yaw Left")
        condition_yaw(45, relative=True, direction=-1)

    elif text == "f":
        print("Executing: Yaw Right")
        condition_yaw(45, relative=True, direction=1)

    elif text == "up":
        print("Executing: Ascend")
        send_ned_velocity(0, 0, -1.0, 2)

    elif text == "down":
        print("Executing: Descend")
        send_ned_velocity(0, 0, 1.0, 2)

    elif text in ["land", "rtl", "return"]:
        print("Landing sequence initiated...")
        set_mode_confirmed("RTL")


def voicecontrol():
    with microphone as source:
        print("Calibrating microphone for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("\n[OFFLINE READY] Awaiting vocal command paths...")

        rec = KaldiRecognizer(vosk_model, 16000)

        while True:
            try:
                audio = recognizer.listen(source, timeout=4, phrase_time_limit=3)
                raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)

                text_input = ""
                if rec.AcceptWaveform(raw_data):
                    result = json.loads(rec.Result())
                    text_input = result.get("text", "").strip().lower()
                else:
                    partial_res = json.loads(rec.FinalResult())
                    text_input = partial_res.get("text", "").strip().lower()

                if text_input:
                    print(f"Speech Recognized (Offline): '{text_input}'")
                    evaluate_voice_command(text_input)

            except sr.WaitTimeoutError:
                continue
            except Exception as error:
                print(f"Audio interface exception: {str(error)}")


# NOTE: the standalone _statustext_listener() thread from the previous version
# has been REMOVED — it duplicated recv_match() calls on the same connection
# as _reader_loop(), reintroducing the exact packet race we fixed earlier.
# STATUSTEXT is already handled centrally inside _reader_loop().


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    try:
        print("\nFailsafe parameter check (helps diagnose unexpected RTL/Land):")
        for p in ["FS_GCS_ENABLE", "FS_THR_ENABLE", "FS_EKF_ACTION",
                  "BATT_FS_LOW_ACT", "BATT_FS_CRT_ACT", "FENCE_ENABLE"]:
            print(f"  {p} = {get_param(p)}")

        arm_and_takeoff(2.0)

        print("\nHovering — voice control is now active. Just speak a command.")
        voice_thread = threading.Thread(target=voicecontrol, daemon=True)
        voice_thread.start()
        voice_thread.join()

    except KeyboardInterrupt:
        print("\nInterrupted by user — landing.")
    except Exception as e:
        print(f"Unexpected error: {e} — landing.")
    finally:
        set_mode_confirmed("RTL")
