import argparse
import sys
import threading
import time
from typing import Optional, Tuple

try:
    import cv2
except ModuleNotFoundError as exc:
    print("Error: OpenCV is not installed in the current Python environment.")
    print("Activate your virtual environment and install dependencies before running this script.")
    print("Example:")
    print("  e:\\python\\gesture_env\\Scripts\\Activate.ps1")
    print("  python main.py")
    raise exc

from gesture_control import GestureBuffer, GESTURE_LABELS, HandGestureDetector
import voice_control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hand gesture detector for drone command mapping."
    )
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=480, help="Capture width (default is a lighter 480p setting for smoother gesture control)")
    parser.add_argument("--height", type=int, default=360, help="Capture height (default is a lighter 480x360 setting for smoother gesture control)")
    parser.add_argument(
        "--buffer-len",
        type=int,
        default=10,
        help="Number of frames used to stabilize gesture output",
    )
    parser.add_argument(
        "--roi-x",
        type=int,
        default=10,
        help="Top-left X coordinate of the tracked region of interest",
    )
    parser.add_argument(
        "--roi-y",
        type=int,
        default=80,
        help="Top-left Y coordinate of the tracked region of interest",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=260,
        help="Width and height of the square ROI used for hand tracking. Smaller ROI keeps the video loop light.",
    )
    parser.add_argument(
        "--mavlink",
        type=str,
        default="udp:192.168.2.1:14550",
        help="MAVLink connection string (serial port, udp, tcp, etc.). Default targets the DroneBridge endpoint at udp:192.168.2.1:14550. Leave empty to disable MAVLink.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=57600,
        help="Baud rate for MAVLink serial connections.",
    )
    parser.add_argument(
        "--takeoff-alt",
        type=float,
        default=4.0,
        help="Target takeoff/hover altitude in meters before control begins.",
    )
    return parser.parse_args()


def draw_status(image, fps: float) -> None:
    """Draw camera performance status on screen."""
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(image, "Press 'q' to quit.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)


def draw_gesture_feed(image, gesture_label: str) -> None:
    """Render a single stable gesture label in the center of the frame."""
    if gesture_label == "None":
        return
    label_text = f"Gesture: {gesture_label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    text_size, _ = cv2.getTextSize(label_text, font, scale, thickness)
    x = (image.shape[1] - text_size[0]) // 2
    y = text_size[1] + 20
    cv2.rectangle(
        image,
        (x - 10, y - text_size[1] - 10),
        (x + text_size[0] + 10, y + 10),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(image, label_text, (x, y), font, scale, (255, 255, 255), thickness)


def draw_mavlink_status(image, status_text: str) -> Tuple[int, int, int, int]:
    """Draw the current MAVLink connection status on the video frame and return the text box rect."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    text_size, _ = cv2.getTextSize(status_text, font, scale, thickness)
    x = 10
    y = image.shape[0] - 20
    rect = (x - 8, y - text_size[1] - 8, x + text_size[0] + 8, y + 8)
    cv2.rectangle(
        image,
        (rect[0], rect[1]),
        (rect[2], rect[3]),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(image, status_text, (x, y), font, scale, (0, 255, 0), thickness)
    return rect


def draw_control_mode_status(image, mode_text: str) -> Tuple[int, int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.6
    hint_scale = 0.45
    thickness = 2
    title = f"Mode: {mode_text.title()}"
    hint = "Click to toggle"
    title_size, _ = cv2.getTextSize(title, font, title_scale, thickness)
    hint_size, _ = cv2.getTextSize(hint, font, hint_scale, 1)
    padding = 10
    gap = 4
    box_width = max(title_size[0], hint_size[0]) + padding * 2
    box_height = title_size[1] + hint_size[1] + gap + padding * 2
    x = image.shape[1] - box_width - 10
    y = 18
    rect = (x, y, x + box_width, y + box_height)
    bg_color = (0, 180, 0) if mode_text == "gesture" else (0, 165, 255)
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), bg_color, cv2.FILLED)
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), 2)
    cv2.putText(image, title, (x + padding, y + padding + title_size[1]), font, title_scale, (255, 255, 255), thickness)
    cv2.putText(
        image,
        hint,
        (x + padding, y + padding + title_size[1] + gap + hint_size[1]),
        font,
        hint_scale,
        (255, 255, 255),
        1,
    )
    return rect


def draw_voice_toggle_status(image, enabled: bool) -> Tuple[int, int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    text = "ShowVoice: ON" if enabled else "ShowVoice: OFF"
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = image.shape[1] - text_size[0] - 10
    y = 60
    rect = (x - 8, y - text_size[1] - 8, x + text_size[0] + 8, y + 8)
    color = (0, 180, 0) if enabled else (80, 80, 80)
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text, (x, y), font, scale, color, thickness)
    return rect


def draw_vehicle_state(image, mode_text: str, armed: bool) -> None:
    """Show the current MAVLink flight mode and arm state in a compact overlay."""
    mode_value = (mode_text or "UNKNOWN").upper()
    armed_text = "ARMED" if armed else "DISARMED"
    armed_color = (0, 255, 0) if armed else (0, 165, 255)

    mode_text_line = f"MODE: {mode_value}"
    state_text_line = f"ARM: {armed_text}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2

    lines = [mode_text_line, state_text_line]
    max_width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)
    x = image.shape[1] - max_width - 20
    y = 110

    bg_x1, bg_y1 = x - 12, y - 26
    bg_x2, bg_y2 = x + max_width + 12, y + 40
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), cv2.FILLED)

    cv2.putText(image, mode_text_line, (x, y), font, scale, (255, 255, 255), thickness)
    cv2.putText(image, state_text_line, (x, y + 24), font, scale, armed_color, thickness)


def draw_voice_feed(image, text: str) -> None:
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = (image.shape[1] - text_size[0]) // 2
    y = image.shape[0] - 60
    cv2.rectangle(image, (x - 10, y - text_size[1] - 10), (x + text_size[0] + 10, y + 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text, (x, y), font, scale, (255, 255, 255), thickness)


def draw_hover_status(image, hover_ready: bool, altitude: float) -> None:
    if not hover_ready:
        text = "Waiting for stable hover..."
        color = (0, 165, 255)
    else:
        text = f"Hover stable: {altitude:.2f} m"
        color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = 10
    y = image.shape[0] - 50
    cv2.rectangle(image, (x - 8, y - text_size[1] - 8), (x + text_size[0] + 8, y + 8), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text, (x, y), font, scale, color, thickness)


def is_point_in_rect(point_x: int, point_y: int, rect: Tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = rect
    return left <= point_x <= right and top <= point_y <= bottom


def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    detector = HandGestureDetector()
    buffer = GestureBuffer(buffer_len=args.buffer_len)

    mav_controller: Optional[object] = None
    mav_controller_lock = threading.Lock()
    voice_controller = None
    stop_event = threading.Event()
    connection_thread = None

    mavlink_target = args.mavlink.strip()
    mavlink_enabled = False
    mav_status = "MAVLink: Disabled" if not mavlink_target else "MAVLink: Disabled"

    mavlink_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    control_mode = "guided"
    last_voice_text = ""
    last_voice_time = 0.0
    VOICE_DISPLAY_TIMEOUT = 3.0
    show_voice_overlay = True
    hover_ready = False
    hover_altitude = 0.0
    vehicle_mode = "UNKNOWN"
    vehicle_armed = False

    def toggle_mavlink() -> None:
        nonlocal mavlink_enabled, mav_status, mav_controller
        if not mavlink_target:
            mav_status = "No MAVLink target configured"
            return

        mavlink_enabled = not mavlink_enabled
        if not mavlink_enabled:
            mav_status = "MAVLink: Disabled"
            with mav_controller_lock:
                if mav_controller is not None:
                    try:
                        mav_controller.close()
                    except Exception:
                        pass
                    mav_controller = None
        else:
            mav_status = "Trying to connect to MAVLink..."

    def on_recognize(text: str) -> None:
        nonlocal last_voice_text, last_voice_time
        last_voice_text = text
        last_voice_time = time.time()

    def on_mouse(event, x, y, flags, param) -> None:
        nonlocal mav_controller, voice_controller, control_mode, voice_toggle_rect, show_voice_overlay
        if event != cv2.EVENT_LBUTTONUP:
            return
        if is_point_in_rect(x, y, mavlink_rect):
            toggle_mavlink()
            return
        if is_point_in_rect(x, y, control_rect):
            # flip mode between gesture and voice (mutually exclusive)
            if control_mode == "gesture":
                # enable voice mode (voice can run without MAVLink controller)
                try:
                    if voice_controller is None:
                        voice_controller = voice_control.VoiceControl(mav_controller, on_recognize=on_recognize)
                        voice_controller.start()
                        print("VoiceControl: started (controller may be None until MAVLink connects).")
                except Exception as e:
                    print(f"VoiceControl start failed: {e}")
                    return
                control_mode = "voice"
                # neutralize any ongoing MAVLink gesture when switching modes
                with mav_controller_lock:
                    if mav_controller is not None:
                        try:
                            mav_controller.update_gesture(-1)
                        except Exception:
                            try:
                                mav_controller.send_neutral()
                            except Exception:
                                pass
            else:
                # switch to gesture: stop voice controller
                try:
                    if voice_controller is not None:
                        voice_controller.close()
                except Exception:
                    pass
                voice_controller = None
                control_mode = "gesture"
                # neutralize when switching back to gesture to avoid stuck command
                with mav_controller_lock:
                    if mav_controller is not None:
                        try:
                            mav_controller.update_gesture(-1)
                        except Exception:
                            try:
                                mav_controller.send_neutral()
                            except Exception:
                                pass

            print(f"Control mode switched to: {control_mode}")
            return

        # voice overlay toggle click
        if is_point_in_rect(x, y, voice_toggle_rect):
            show_voice_overlay = not show_voice_overlay
            print(f"Show voice overlay: {show_voice_overlay}")

    cv2.namedWindow("Aerosign Gesture Control", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Aerosign Gesture Control", on_mouse)

    if mavlink_target:

        def connection_loop() -> None:
            nonlocal mav_controller, mav_status, voice_controller, hover_ready, hover_altitude
            while not stop_event.is_set():
                with mav_controller_lock:
                    if not mavlink_enabled:
                        if mav_controller is not None:
                            try:
                                mav_controller.close()
                            except Exception:
                                pass
                            mav_controller = None
                            try:
                                if voice_controller is not None:
                                    voice_controller.set_controller(None)
                            except Exception:
                                pass
                        mav_status = "MAVLink: Disabled"
                    elif mav_controller is None:
                        mav_status = "Trying to connect to MAVLink..."
                        try:
                            from mavlink_control import MavlinkGestureController
                            controller = MavlinkGestureController(
                                connection_string=mavlink_target,
                                baud=args.baud,
                            )
                            mav_controller = controller
                            mav_status = "MAVLink connected"
                            print(f"MAVLink: connected to {mavlink_target} @ {args.baud}")

                            # Safe startup sequence for outdoor flight: GUIDED mode, arming,
                            # takeoff, and stable hover before gesture control is allowed.
                            try:
                                hover_ok, hover_note = controller.safe_startup(args.takeoff_alt)
                                if hover_ok:
                                    hover_ready = True
                                    mav_status = f"Hover stable: {hover_note}"
                                    print(f"[AUTO-HOVER] {hover_note}")
                                    controller.update_gesture(-1)
                                else:
                                    hover_ready = False
                                    mav_status = f"MAVLink connected; startup failed: {hover_note}"
                                    print(f"[AUTO-HOVER] {hover_note}")
                            except Exception as exc:
                                print(f"[MAVLINK] Guided startup failed: {exc}")
                                hover_ready = False
                                try:
                                    controller.trigger_rtl("Startup failure")
                                except Exception:
                                    pass
                                mav_status = "MAVLink connected; guided startup failed"

                            # bind running voice controller (if any) to this controller and start listening automatically
                            try:
                                if voice_controller is None:
                                    voice_controller = voice_control.VoiceControl(mav_controller, on_recognize=on_recognize)
                                    voice_controller.start()
                                    print("VoiceControl: started automatically after MAVLink connection.")
                                else:
                                    voice_controller.set_controller(mav_controller)
                            except Exception:
                                pass
                        except ModuleNotFoundError as exc:
                            mav_status = "pymavlink not installed"
                            print(f"MAVLink error: {exc}")
                            return
                        except Exception as exc:
                            mav_status = "MAVLink connection failed"
                            print(f"MAVLink connection failed: {exc}")
                stop_event.wait(2.0)

        connection_thread = threading.Thread(target=connection_loop, daemon=True)
        connection_thread.start()

    prev_time = time.time()
    last_display_label = "None"
    last_mav_gesture = -1
    stable_timeout = 0.7
    last_stable_time = time.time()
    last_state_poll = time.time()
    target_fps = 20.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read error.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        x1 = max(0, min(args.roi_x, w - args.roi_size))
        y1 = max(0, min(args.roi_y, h - args.roi_size))
        x2 = min(w, x1 + args.roi_size)
        y2 = min(h, y1 + args.roi_size)

        roi = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        now = time.time()

        if control_mode == "gesture":
            processed_roi, gesture_id = detector.process(roi)
            buffer.add_gesture(gesture_id)
            command_id = buffer.get_gesture()
        else:
            # Don't process gestures in voice mode
            processed_roi = roi
            gesture_id = -1
            command_id = -1
        gesture_label = GESTURE_LABELS.get(command_id, "None")

        if command_id != -1:
            last_display_label = gesture_label
            last_stable_time = now
        elif now - last_stable_time > stable_timeout:
            last_display_label = "None"

        # Send commands only from the active control mode
        with mav_controller_lock:
            controller = mav_controller
        if controller is not None:
            try:
                if now - last_state_poll >= 0.2:
                    last_state_poll = now
                    current_mode = controller.get_mode() if hasattr(controller, "get_mode") else "UNKNOWN"
                    current_armed = controller.is_armed() if hasattr(controller, "is_armed") else False
                else:
                    current_mode = vehicle_mode
                    current_armed = vehicle_armed

                if not controller.ensure_safe_state():
                    hover_ready = False
                    mav_status = "Safety abort: RTL engaged"
                    try:
                        controller.update_gesture(-1)
                    except Exception:
                        pass
                    last_mav_gesture = -1
                    raise RuntimeError("Vehicle not in a safe state")

                if not hover_ready:
                    if time.time() - last_stable_time > 0.5:
                        print(f"[MAVLINK CHECK] Waiting for stable hover. Current mode={current_mode}, armed={current_armed}")
                    last_mav_gesture = -1
                    try:
                        controller.update_gesture(-1)
                    except Exception:
                        pass
                elif control_mode == "gesture":
                    if command_id != -1 and command_id != last_mav_gesture:
                        try:
                            print(f"[MAVLINK DEBUG] sending gesture {command_id} to rc override")
                            controller.update_gesture(command_id)
                            last_mav_gesture = command_id
                        except Exception as exc:
                            print(f"[MAVLINK DEBUG] send failed: {exc}")
                            with mav_controller_lock:
                                mav_controller = None
                                mav_status = "Trying to connect to MAVLink..."
                            last_mav_gesture = -1
                    elif command_id == -1 and last_mav_gesture != -1:
                        try:
                            print("[MAVLINK DEBUG] sending neutral RC")
                            controller.update_gesture(-1)
                            last_mav_gesture = -1
                        except Exception as exc:
                            print(f"[MAVLINK DEBUG] neutral send failed: {exc}")
                            with mav_controller_lock:
                                mav_controller = None
                                mav_status = "Trying to connect to MAVLink..."
                            last_mav_gesture = -1
            except Exception as exc:
                print(f"[MAVLINK CHECK] State validation failed: {exc}")

        dt = now - prev_time
        fps = 1.0 / dt if dt > 0 else 0.0
        prev_time = now

        # Keep the control loop from running too fast when the camera is already limited.
        if dt < 1.0 / target_fps:
            time.sleep(0.005)

        # copy the processed ROI back into the frame for display
        frame[y1:y2, x1:x2] = processed_roi

        # draw gesture label only when gesture mode is active
        if control_mode == "gesture":
            draw_gesture_feed(frame, last_display_label)
        draw_status(frame, fps)
        with mav_controller_lock:
            current_status = mav_status
            controller = mav_controller
        if controller is not None and now - last_state_poll >= 0.2:
            last_state_poll = now
            try:
                vehicle_mode = controller.get_mode() if hasattr(controller, "get_mode") else "UNKNOWN"
                vehicle_armed = controller.is_armed() if hasattr(controller, "is_armed") else False
            except Exception:
                vehicle_mode = "UNKNOWN"
                vehicle_armed = False
        elif controller is None:
            vehicle_mode = "UNKNOWN"
            vehicle_armed = False
        mavlink_rect = draw_mavlink_status(frame, current_status)
        control_rect = draw_control_mode_status(frame, control_mode)
        voice_toggle_rect = draw_voice_toggle_status(frame, show_voice_overlay)
        draw_vehicle_state(frame, vehicle_mode, vehicle_armed)
        draw_hover_status(frame, hover_ready, hover_altitude)

        # show voice/gesture overlay for the active mode (if enabled)
        if show_voice_overlay and (time.time() - last_voice_time < VOICE_DISPLAY_TIMEOUT):
            draw_voice_feed(frame, last_voice_text)

        cv2.imshow("Aerosign Gesture Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord('v'):
            show_voice_overlay = not show_voice_overlay
            print(f"Show voice overlay toggled: {show_voice_overlay}")

    detector.close()
    stop_event.set()
    if connection_thread is not None:
        connection_thread.join(timeout=1.0)
    with mav_controller_lock:
        controller = mav_controller
    if controller is not None:
        try:
            controller.close()
        except Exception:
            pass
    # close voice controller if running
    try:
        if voice_controller is not None:
            voice_controller.close()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
