import threading
import time
import socket
import json
from pathlib import Path
import speech_recognition as sr
try:
    from vosk import Model, KaldiRecognizer  # optional dependency for offline recognition
except Exception:
    Model = None
    KaldiRecognizer = None


class VoiceControl:
    """Simple voice-to-gesture bridge using SpeechRecognition.

    This expects a controller object that implements `update_gesture(gesture_id)`
    and `send_land_command()` (matching `MavlinkGestureController`).

    Usage: create instance, then call `start()` to begin listening and
    `close()` to stop. Provide `on_recognize` callback to receive recognized
    text for UI display.
    """

    def __init__(self, controller=None, on_recognize=None):
        self.controller = controller
        self.on_recognize = on_recognize
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self._stop = threading.Event()
        self._thread = None
        # Try to load the offline Vosk model from the project root or legacy model locations.
        # The model directory itself must be the extracted Vosk folder, not the parent directory
        # that contains several subfolders like "vosk-model-small-en-us-0.15".
        self.vosk_model = None
        self._vosk_available = False

        candidate_roots = [
            Path(__file__).parent / "model",
            Path(__file__).parent / "models" / "vosk",
            Path.cwd() / "model",
            Path.cwd() / "models" / "vosk",
        ]
        candidate_paths = []
        for base in candidate_roots:
            if not base.exists():
                continue
            if (base / "graph").exists() or (base / "conf").exists():
                candidate_paths.append(base)
            else:
                for child in sorted(base.iterdir()):
                    if child.is_dir() and ((child / "graph").exists() or (child / "conf").exists()):
                        candidate_paths.append(child)

        found_model = False
        for default_model in candidate_paths:
            if Model is None:
                continue
            found_model = True
            try:
                self.vosk_model = Model(str(default_model))
                self._vosk_available = True
                print(f"VoiceControl: Vosk model loaded from {default_model}")
                break
            except Exception as e:
                print(f"VoiceControl: failed to load Vosk model from {default_model}: {e}")

        if not found_model:
            print("VoiceControl: no Vosk model folder found.")
            print("VoiceControl: to enable offline voice recognition, download a model such as")
            print("VoiceControl:   vosk-model-small-en-us-0.15")
            print("VoiceControl: and extract it into a folder named 'model' beside main.py")
            print("VoiceControl: example: E:\\AEROSIGN\\aerosign-gesture-mavlink-master\\model\\vosk-model-small-en-us-0.15")
            print("VoiceControl: without the model, recognition will only work with an internet-enabled Google recognizer.")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def set_controller(self, controller):
        """Set or replace the underlying MAV controller while running."""
        self.controller = controller

    def _listen_loop(self):
        try:
            with self.microphone as source:
                # calibrate once for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("VoiceControl: Microphone calibrated. Awaiting commands...")

                while not self._stop.is_set():
                    try:
                        audio = self.recognizer.listen(source, timeout=4, phrase_time_limit=3)
                        text = ""
                        # Prefer offline Vosk if available (no internet required)
                        if self._vosk_available:
                            try:
                                SAMPLE_RATE = 16000
                                raw = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=2)
                                rec = KaldiRecognizer(self.vosk_model, SAMPLE_RATE)
                                rec.AcceptWaveform(raw)
                                final = rec.FinalResult()
                                try:
                                    res = json.loads(final)
                                    text = res.get("text", "") or ""
                                except Exception:
                                    text = ""
                            except Exception as exc:
                                print(f"[VoiceControl] Vosk recognition error: {exc}")
                                text = ""
                        # Fallback to Google online recognizer
                        if not text:
                            try:
                                text = self.recognizer.recognize_google(audio).lower()
                            except Exception as exc:
                                print(f"[VoiceControl] Online recognizer failed: {exc}")
                                # try SR's recognize_vosk if built-in and model present
                                try:
                                    text = self.recognizer.recognize_vosk(audio).lower()
                                except Exception as exc2:
                                    print(f"[VoiceControl] Offline recognizer not available or failed: {exc2}")
                                    time.sleep(0.5)
                                    continue

                        if text:
                            print(f"VoiceControl recognized: '{text}'")
                            if callable(self.on_recognize):
                                try:
                                    self.on_recognize(text)
                                except Exception:
                                    pass
                            self._evaluate_text(text)
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        # couldn't understand phrase
                        continue
                    except Exception as exc:
                        print(f"[VoiceControl] Audio exception: {exc}")
                        time.sleep(0.5)
        except Exception as exc:
            print(f"[VoiceControl] Failed to open microphone: {exc}")

    def _evaluate_text(self, text: str) -> None:
        t = text.lower()
        # Map simple vocal commands to gesture ids used by MavlinkGestureController
        if "land" in t and "land" == t.strip():
            # explicit short 'land' -> use land command if controller available
            if self.controller is None:
                print("VoiceControl: land requested but no MAV controller bound.")
                return
            try:
                self.controller.send_land_command()
            except Exception as e:
                print(f"VoiceControl: send_land_command failed: {e}")
            return

        # If no controller is bound, only announce recognized commands
        if self.controller is None:
            print(f"VoiceControl: recognized '{t}' (no MAV controller bound)")
            return

        if "forward" in t or "go" in t:
            print("VoiceControl: mapped to FORWARD (0) for 2.0s")
            self.controller.update_gesture(0, duration=2.0)
        elif "back" in t:
            print("VoiceControl: mapped to BACK (1) for 2.0s")
            self.controller.update_gesture(1, duration=2.0)
        elif "up" in t or "takeoff" in t:
            print("VoiceControl: mapped to UP (2) for 2.0s")
            self.controller.update_gesture(2, duration=2.0)
        elif "down" in t:
            print("VoiceControl: mapped to DOWN (3) for 2.0s")
            self.controller.update_gesture(3, duration=2.0)
        elif "left" in t:
            print("VoiceControl: mapped to LEFT (4) for 2.0s")
            self.controller.update_gesture(4, duration=2.0)
        elif "right" in t:
            print("VoiceControl: mapped to RIGHT (5) for 2.0s")
            self.controller.update_gesture(5, duration=2.0)
        elif "hover" in t or "stop" in t or "hold" in t:
            print("VoiceControl: mapped to HOVER (6)")
            self.controller.update_gesture(6)
        elif "rtl" in t or "return" in t:
            try:
                if hasattr(self.controller, "trigger_rtl"):
                    self.controller.trigger_rtl("Voice command")
                else:
                    self.controller.send_rtl_command()
            except Exception as e:
                print(f"VoiceControl: RTL/return command failed: {e}")

    def close(self, timeout: float = 1.0) -> None:
        self._stop.set()
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=timeout)
        except Exception:
            pass
