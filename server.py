#!/usr/bin/env python3
"""
server.py - Bidirectional Optical Data Transfer (Server / Local Sender)
Acts as the file sender and optical telemetry decoder for full-duplex air-gap transfer.

Channels:
- Visual / Optical (Client -> Server): Captures client screen/window via browser
  PipeWire screen share with optical ROI mask; decodes corner fiducials and 32x32 color grid
  to monitor client focus state and verify chunk reception (ACK bitmask + SHA-256).
- Keyboard (Server -> Client): Types standard alphanumeric packets (hex framing + CRC32)
  directly into the focused client window using low-latency keystroke emulation.

Workflow:
1. Start server.py (launches web interface at http://localhost:8080).
2. In the browser, click "Share Screen/Window" and select the client.py window.
3. Draw or Auto-Snap the transmission mask around the client's optical grid.
4. Select a file to send via the browser file picker or drag-and-drop.
5. Focus the client.py window and press ENTER or SPACE.
6. Server detects focus via the client's optical grid and automatically types the file!
7. Verification occurs in real-time as the client ACKs received chunks on its grid.
"""

import argparse
import base64
import hashlib
import http.server
import json
import os
import socket
import socketserver
import struct
import sys
import threading
import time
import webbrowser
import zlib
import cv2
import numpy as np

# Suppress Qt font lookup warnings
os.environ["QT_LOGGING_RULES"] = "*.warning=false"

from decoder_v2 import FiducialTracker

# Client states decoded from optical grid
STATE_WAIT_FOCUS = 0
STATE_FOCUSED = 1
STATE_RECEIVING = 2
STATE_COMPLETE = 3
STATE_ERROR = 4

STATE_NAMES = {
    0: "WAITING_FOR_FOCUS",
    1: "KEYBOARD_FOCUSED",
    2: "RECEIVING_DATA",
    3: "VERIFIED_COMPLETE",
    4: "TRANSFER_ERROR"
}


# ==============================================================================
# KEYBOARD EMULATION ENGINE (Alphanumeric Only)
# ==============================================================================

class KeyboardTyper:
    """
    Cross-platform keystroke synthesizer strictly emitting standard alphanumeric
    characters [0-9a-zA-Z] and newline (Return).
    Immune to international keyboard layout symbol permutations.
    """
    def __init__(self, char_delay=0.002):
        self.char_delay = char_delay
        self.backend = None
        self._init_backend()

    def _init_backend(self):
        # 1. Primary on Linux: python-xlib with Mutter xauth patch
        try:
            from Xlib import X, XK, xauth, display
            from Xlib.ext import xtest

            # Mutter / GNOME xauth patch for Xwayland
            orig_get = xauth.Xauthority.get_best_auth
            def patched_get_best(s, family, address, dispno, types=(b"MIT-MAGIC-COOKIE-1",)):
                try:
                    return orig_get(s, family, address, dispno, types)
                except Exception:
                    matches = {}
                    for efam, eaddr, enum, ename, edata in s.entries:
                        if (efam == family or efam == 65535) and (enum == b"" or enum == str(dispno).encode()):
                            matches[ename] = edata
                    for t in types:
                        if t in matches:
                            return (t, matches[t])
                    raise
            xauth.Xauthority.get_best_auth = patched_get_best

            disp = display.Display()
            if disp.has_extension("XTEST"):
                self.disp = disp
                self.X = X
                self.XK = XK
                self.xtest = xtest
                self.backend = "xlib"
                return
        except Exception:
            pass

        # 2. Secondary fallback: PyAutoGUI if available
        try:
            import pyautogui
            self.pyautogui = pyautogui
            self.backend = "pyautogui"
            return
        except Exception:
            pass

        # 3. Fallback: Dummy logger
        self.backend = "stdout_fallback"

    def is_available(self):
        return self.backend in ("xlib", "pyautogui")

    def request_access(self):
        """
        Sends a harmless Shift press/release via XTest to prompt the OS/compositor
        (e.g., GNOME Mutter on Wayland) to display the system permission prompt
        upfront before transmission begins.
        """
        if self.backend == "xlib":
            try:
                shift_kc = self.disp.keysym_to_keycode(self.XK.string_to_keysym("Shift_L"))
                if shift_kc:
                    self.xtest.fake_input(self.disp, self.X.KeyPress, shift_kc)
                    self.xtest.fake_input(self.disp, self.X.KeyRelease, shift_kc)
                    self.disp.sync()
                    return True
            except Exception as e:
                print(f"[Typer] Error requesting keyboard access: {e}")
                return False
        return False

    def type_char(self, ch, sync=True):
        if self.backend == "xlib":
            try:
                if ch in ("\n", "\r"):
                    ks = self.XK.string_to_keysym("Return")
                    needs_shift = False
                elif ch == " ":
                    ks = self.XK.string_to_keysym("space")
                    needs_shift = False
                else:
                    ks = self.XK.string_to_keysym(ch)
                    needs_shift = ch.isupper()

                if not ks:
                    return False
                kc = self.disp.keysym_to_keycode(ks)
                if not kc:
                    return False

                shift_kc = self.disp.keysym_to_keycode(self.XK.string_to_keysym("Shift_L")) if needs_shift else None
                if needs_shift and shift_kc:
                    self.xtest.fake_input(self.disp, self.X.KeyPress, shift_kc)
                self.xtest.fake_input(self.disp, self.X.KeyPress, kc)
                self.xtest.fake_input(self.disp, self.X.KeyRelease, kc)
                if needs_shift and shift_kc:
                    self.xtest.fake_input(self.disp, self.X.KeyRelease, shift_kc)
                if sync:
                    self.disp.sync()
                return True
            except Exception:
                return False

        elif self.backend == "pyautogui":
            try:
                if ch in ("\n", "\r"):
                    self.pyautogui.press("enter")
                else:
                    self.pyautogui.write(ch)
                return True
            except Exception:
                return False

        elif self.backend == "stdout_fallback":
            sys.stdout.write(ch)
            sys.stdout.flush()
            return True

        return False

    def release_all_keys(self):
        """
        Releases all modifier and standard keys to guarantee no keys remain stuck down
        when transmission finishes or is aborted.
        """
        if self.backend == "xlib":
            try:
                keys = ["Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Super_L", "Return", "space"]
                for k in keys:
                    ks = self.XK.string_to_keysym(k)
                    if ks:
                        kc = self.disp.keysym_to_keycode(ks)
                        if kc:
                            self.xtest.fake_input(self.disp, self.X.KeyRelease, kc)
                self.disp.sync()
            except Exception:
                pass

    def type_string(self, text, delay=None, cancel_check=None):
        cd = self.char_delay if delay is None else delay
        if self.backend == "xlib":
            try:
                count = 0
                for ch in text:
                    if cancel_check and (count % 8 == 0) and cancel_check():
                        self.release_all_keys()
                        return False
                    count += 1
                    self.type_char(ch, sync=False)
                    if count % 16 == 0:
                        self.disp.sync()
                    if cd > 0:
                        time.sleep(cd)
                if cancel_check and cancel_check():
                    self.release_all_keys()
                    return False
                self.disp.sync()
                return True
            except Exception:
                pass

        count = 0
        for ch in text:
            if cancel_check and (count % 8 == 0) and cancel_check():
                self.release_all_keys()
                return False
            count += 1
            self.type_char(ch, sync=True)
            if cd > 0:
                time.sleep(cd)
        return True


# ==============================================================================
# PACKET PROTOCOL BUILDERS
# ==============================================================================

def make_header_packet(raw_size, comp_size, total_chunks, sha256_hex, filename):
    fn_bytes = filename.encode("utf-8")
    fn_len = len(fn_bytes)
    fn_hex = fn_bytes.hex()
    body = f"{raw_size:08x}{comp_size:08x}{total_chunks:04x}{sha256_hex}{fn_len:02x}{fn_hex}"
    crc = zlib.crc32(body.encode("ascii")) & 0xffffffff
    return f"h{body}{crc:08x}\n"


def make_data_packet(idx, payload_bytes):
    crc = zlib.crc32(payload_bytes) & 0xffffffff
    plen = len(payload_bytes)
    phex = payload_bytes.hex()
    return f"d{idx:04x}{crc:08x}{plen:04x}{phex}\n"


def make_chat_packet(msg_str):
    msg_bytes = msg_str.encode("utf-8")
    msg_len = len(msg_bytes)
    msg_hex = msg_bytes.hex()
    body = f"{msg_len:04x}{msg_hex}"
    crc = zlib.crc32(body.encode("ascii")) & 0xffffffff
    return f"m{body}{crc:08x}\n"


def make_end_packet(sha256_hex):
    crc = zlib.crc32(sha256_hex.encode("ascii")) & 0xffffffff
    return f"e{sha256_hex}{crc:08x}\n"


def unpack_optical_ack_payload(payload):
    """
    Decodes the structured ACK2 optical payload emitted by client.py.
    """
    if len(payload) < 118:
        return None
    body = payload[:114]
    crc_expected = struct.unpack(">I", payload[114:118])[0]
    if (zlib.crc32(body) & 0xffffffff) != crc_expected:
        return None

    magic, state, last_ack, tot, cnt, b_rec, mask, sha, err = struct.unpack(">4sBHHII64s32sB", body)
    if magic != b"ACK2":
        return None

    focus_seq = (err >> 4) & 0x0F
    err_code = err & 0x0F

    return {
        "state": state,
        "state_name": STATE_NAMES.get(state, f"UNKNOWN_{state}"),
        "last_ack": last_ack,
        "total_chunks": tot,
        "chunks_count": cnt,
        "bytes_received": b_rec,
        "bitmask": mask,
        "sha256": sha.hex(),
        "error_code": err_code,
        "focus_seq": focus_seq
    }


# ==============================================================================
# SERVER STATE & OPTICAL TRACKING SESSION
# ==============================================================================

class ServerDuplexSession:
    def __init__(self, chunk_size=256, char_delay=0.0002):
        self.chunk_size = chunk_size
        self.char_delay = char_delay
        self.packet_delay = 0.008
        self.speed_preset = "fast"
        self.tracker = FiducialTracker()
        self.typer = KeyboardTyper(char_delay=char_delay)

        self.lock = threading.RLock()

        # Optical link state
        self.optical_locked = False
        self.last_fiducials = None
        self.client_state = STATE_WAIT_FOCUS
        self.client_focus_seq = 0
        self.client_last_ack = 0
        self.client_total_chunks = 0
        self.client_chunks_count = 0
        self.client_bytes_received = 0
        self.client_bitmask = bytearray(64)
        self.client_sha = ""
        self.client_error = 0
        self.last_optical_frame_time = 0
        self.last_optical_decode_time = 0

        # File staging & transmission state
        self.staged_file = None      # dict: name, raw_size, comp_size, sha256, chunks, packets, raw_bytes
        self.transfer_active = False
        self.transfer_complete = False
        self.transfer_error = None
        self.abort_requested = False
        self.chunks_sent = 0
        self.chunks_acked = 0
        self.retransmissions = 0
        self.start_time = 0
        self.end_time = 0

        self.focus_confirmed_event = threading.Event()
        self.activity_log = []

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        with self.lock:
            self.activity_log.append(entry)
            if len(self.activity_log) > 50:
                self.activity_log.pop(0)
        print(f"[Server] {entry}")

    def set_speed_preset(self, preset):
        with self.lock:
            self.speed_preset = preset
            if preset == "turbo":
                self.chunk_size = 256
                self.char_delay = 0.0
                self.packet_delay = 0.003
            elif preset == "fast":
                self.chunk_size = 256
                self.char_delay = 0.0002
                self.packet_delay = 0.008
            elif preset == "balanced":
                self.chunk_size = 192
                self.char_delay = 0.001
                self.packet_delay = 0.015
            elif preset == "safe":
                self.chunk_size = 128
                self.char_delay = 0.002
                self.packet_delay = 0.025
            self.typer.char_delay = self.char_delay

            # If a file is staged and transfer is not yet running, restage with the new chunk size
            if self.staged_file and not self.transfer_active and "raw_bytes" in self.staged_file:
                raw_bytes = self.staged_file["raw_bytes"]
                filename = self.staged_file["filename"]
                self._restage_unlocked(filename, raw_bytes)

        self.log(f"Speed set to '{preset.upper()}': {self.chunk_size}B/chunk, {self.char_delay*1000:.1f}ms char delay, {self.packet_delay*1000:.0f}ms pkt delay.")

    def _restage_unlocked(self, filename, raw_bytes):
        raw_size = len(raw_bytes)
        comp_bytes = zlib.compress(raw_bytes, 9)
        comp_size = len(comp_bytes)
        sha256_hex = hashlib.sha256(raw_bytes).hexdigest()

        cap = self.chunk_size
        chunks = [comp_bytes[i:i + cap] for i in range(0, len(comp_bytes), cap)]
        tot_chunks = len(chunks)

        header_pkt = make_header_packet(raw_size, comp_size, tot_chunks, sha256_hex, filename)
        data_pkts = [make_data_packet(i, chunks[i]) for i in range(tot_chunks)]
        end_pkt = make_end_packet(sha256_hex)

        self.staged_file = {
            "filename": filename,
            "raw_size": raw_size,
            "comp_size": comp_size,
            "sha256": sha256_hex,
            "chunks": chunks,
            "total_chunks": tot_chunks,
            "header_pkt": header_pkt,
            "data_pkts": data_pkts,
            "end_pkt": end_pkt,
            "acked_chunks": [False] * tot_chunks,
            "raw_bytes": raw_bytes
        }

    def stage_file(self, filename, raw_bytes):
        with self.lock:
            if self.transfer_active:
                return False, "Transfer is currently in progress."

            self._restage_unlocked(filename, raw_bytes)

            self.transfer_active = False
            self.transfer_complete = False
            self.transfer_error = None
            self.abort_requested = False
            self.chunks_sent = 0
            self.chunks_acked = 0
            self.retransmissions = 0
            self.focus_confirmed_event.clear()

        # Proactively request keyboard access via harmless Shift key if not already tested
        self.typer.request_access()

        self.log(f"Staged file: '{filename}' ({len(raw_bytes)} bytes -> {self.staged_file['comp_size']} bytes, {self.staged_file['total_chunks']} chunks).")
        self.log("Action required: Focus the client terminal (or GUI window) and press ENTER or SPACE.")
        return True, "File staged successfully."

    def is_link_alive(self, timeout=2.5):
        """
        Dead-man's switch: returns True only if the optical link is actively locked,
        the client state is healthy, and a fresh optical ACK frame was decoded within timeout seconds (default 2.5s).
        """
        with self.lock:
            if not self.transfer_active or self.abort_requested:
                return False
            if not self.optical_locked:
                return False
            if time.time() - self.last_optical_decode_time > timeout:
                return False
            if self.client_state == STATE_ERROR:
                return False
            return True

    def is_chunk_acked(self, idx):
        """
        Returns True if chunk idx has been verified as received by the client
        via optical bitmask telemetry or last_ack confirmation.
        """
        with self.lock:
            if not self.staged_file:
                return False
            acked = self.staged_file.get("acked_chunks", [])
            if 0 <= idx < len(acked) and acked[idx]:
                return True
            # For chunks beyond the 64-byte bitmask (idx >= 512):
            if idx >= 512 and self.client_chunks_count > idx and self.client_last_ack >= idx:
                if 0 <= idx < len(acked):
                    acked[idx] = True
                return True
            return False

    def abort_transmission(self, reason="Manual or safety stop"):
        """
        Immediately halts keystroke transmission, releases all keyboard keys,
        and marks the transfer aborted.
        """
        with self.lock:
            if not self.transfer_active and not self.staged_file:
                return
            was_active = self.transfer_active
            self.transfer_active = False
            self.abort_requested = True
            self.transfer_error = reason

        self.typer.release_all_keys()
        if was_active:
            self.log(f"[🛑 SAFETY HALT] Keystrokes immediately stopped: {reason}")
        else:
            self.log(f"[🛑 Staged transfer cancelled]: {reason}")

    def process_image_bytes(self, img_bytes):
        """Processes incoming video frame from browser WebSocket."""
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"status": "error", "message": "Failed to decode image frame"}

        # Disable thumbnail diff bypass so every live frame is decoded fresh,
        # allowing instantaneous detection of single-bit optical state transitions.
        self.tracker.prev_thumb = None
        if not self.optical_locked:
            self.tracker.locked_geometry = None

        res, fiducials, _ = self.tracker.decode_frame(frame, locked_grid_size=32)

        with self.lock:
            if res is not None:
                self.optical_locked = True
                self.last_optical_frame_time = time.time()
                self.last_optical_decode_time = time.time()
                if fiducials:
                    self.last_fiducials = [[float(p[0]), float(p[1])] for p in fiducials]

                idx, total, payload, gs, mode = res
                ack_info = unpack_optical_ack_payload(payload)

                if ack_info:
                    old_state = self.client_state
                    old_seq = self.client_focus_seq
                    self.client_state = ack_info["state"]
                    self.client_focus_seq = ack_info.get("focus_seq", 0)
                    self.client_last_ack = ack_info["last_ack"]
                    self.client_total_chunks = ack_info["total_chunks"]
                    self.client_chunks_count = ack_info["chunks_count"]
                    self.client_bytes_received = ack_info["bytes_received"]
                    self.client_bitmask = ack_info["bitmask"]
                    self.client_sha = ack_info["sha256"]
                    self.client_error = ack_info["error_code"]

                    # Check for focus confirmation
                    focus_triggered = (
                        (self.client_state >= STATE_FOCUSED and not self.focus_confirmed_event.is_set()) or
                        (self.client_state >= STATE_FOCUSED and self.client_focus_seq != old_seq)
                    )
                    if focus_triggered:
                        self.focus_confirmed_event.set()
                        st_name = STATE_NAMES.get(self.client_state, str(self.client_state))
                        self.log(f"Client keyboard focus CONFIRMED via optical grid (state={st_name}, seq={self.client_focus_seq})!")

                    # Immediately trigger transmission when client window is focused and file is staged
                    if (
                        self.client_state >= STATE_FOCUSED
                        and self.staged_file
                        and not self.transfer_active
                        and not self.transfer_complete
                        and not self.abort_requested
                    ):
                        self.start_transmission_worker()

                    # Update chunk ACKs from bitmask
                    if self.staged_file and self.transfer_active:
                        acked_list = self.staged_file["acked_chunks"]
                        mask = self.client_bitmask
                        count = 0
                        for i in range(len(acked_list)):
                            b_pos = i // 8
                            b_bit = i % 8
                            if b_pos < len(mask) and (mask[b_pos] & (1 << b_bit)):
                                acked_list[i] = True
                                count += 1
                        self.chunks_acked = count

                    # Check completion confirmation
                    if self.client_state == STATE_COMPLETE and self.transfer_active:
                        self.transfer_active = False
                        self.transfer_complete = True
                        self.end_time = time.time()
                        self.log("Transmission 100% COMPLETE & VERIFIED by client optical grid!")
                        self.typer.release_all_keys()

            else:
                # Optical decode failed on this frame
                if time.time() - self.last_optical_decode_time > 2.5:
                    self.optical_locked = False
                    self.last_fiducials = None
                    if self.transfer_active:
                        self.abort_transmission(reason="Optical signal lost for >2.5s (client window obscured, minimized, or moved)")

            # Compile telemetry response for WebSocket
            progress = 0.0
            tot = 1
            if self.staged_file:
                tot = self.staged_file["total_chunks"]
                progress = (self.chunks_acked / max(1, tot)) * 100.0

            speed = 0.0
            if self.transfer_active and self.start_time > 0:
                dt = time.time() - self.start_time
                if dt > 0.5:
                    speed = (self.chunks_acked * self.chunk_size) / dt

            return {
                "optical_locked": self.optical_locked,
                "fiducials": self.last_fiducials,
                "client_state": STATE_NAMES.get(self.client_state, "UNKNOWN"),
                "client_state_code": self.client_state,
                "focus_confirmed": self.focus_confirmed_event.is_set(),
                "transfer_active": self.transfer_active,
                "transfer_complete": self.transfer_complete,
                "abort_requested": self.abort_requested,
                "chunks_sent": self.chunks_sent,
                "chunks_acked": self.chunks_acked,
                "total_chunks": tot if self.staged_file else 0,
                "progress_pct": round(progress, 1),
                "speed_bps": round(speed, 1),
                "filename": self.staged_file["filename"] if self.staged_file else "",
                "retransmissions": self.retransmissions,
                "typer_backend": self.typer.backend,
                "speed_preset": self.speed_preset
            }

    def start_transmission_worker(self):
        t = threading.Thread(target=self._transmission_worker_loop, daemon=True)
        t.start()

    def _transmission_worker_loop(self):
        with self.lock:
            if not self.staged_file or self.transfer_active or self.transfer_complete:
                return
            self.transfer_active = True
            self.abort_requested = False
            self.start_time = time.time()
            staged = self.staged_file

        self.log(f"Starting keyboard transmission of '{staged['filename']}' ({staged['total_chunks']} chunks)...")
        cancel_header_cb = lambda: not self.is_link_alive(timeout=3.0)
        cancel_data_cb = lambda: not self.is_link_alive(timeout=2.5)

        try:
            # Give client a brief 150ms moment to settle focus
            time.sleep(0.15)

            # 1. Send Header packet
            header_pkt = staged["header_pkt"]
            header_acked = False
            for attempt in range(8):
                if not self.is_link_alive(timeout=3.0):
                    self.abort_transmission("Optical link lost before header acknowledged")
                    return

                self.log(f"Sending header packet (attempt {attempt + 1}/8)...")
                ok = self.typer.type_string(header_pkt, delay=self.char_delay, cancel_check=cancel_header_cb)
                if not ok or not self.is_link_alive(timeout=3.0):
                    self.abort_transmission("Optical link interrupted during header typing")
                    return

                # Wait up to 2.0s for client to enter STATE_RECEIVING
                t_wait = time.time() + 2.0
                while time.time() < t_wait:
                    time.sleep(0.04)
                    if not self.is_link_alive(timeout=3.0):
                        self.abort_transmission("Optical link lost while awaiting header ACK")
                        return
                    with self.lock:
                        if self.client_state == STATE_RECEIVING:
                            header_acked = True
                            break
                if header_acked:
                    self.log("Client acknowledged header packet on optical grid (STATE_RECEIVING)!")
                    break

            if not header_acked:
                self.abort_transmission("Client did not acknowledge header on optical grid after 8 attempts. Halting to avoid typing into wrong window.")
                return

            # 2. Transmit Data chunks with Lock-Step Optical Acknowledgment
            # Every chunk must be acknowledged by the client before transmitting the next.
            tot = staged["total_chunks"]
            max_chunk_retries = 10

            for idx in range(tot):
                # If already acked (e.g. from an earlier frame), advance immediately
                if self.is_chunk_acked(idx):
                    continue

                chunk_acked = False
                for attempt in range(max_chunk_retries):
                    if not self.is_link_alive(timeout=2.5):
                        self.abort_transmission(f"Optical link lost at chunk {idx + 1}/{tot}")
                        return

                    # On retry, send a newline first to flush any partial line in client buffer
                    if attempt > 0:
                        self.typer.type_string("\n")
                        time.sleep(0.02)

                    pkt = staged["data_pkts"][idx]
                    ok = self.typer.type_string(pkt, delay=self.char_delay, cancel_check=cancel_data_cb)
                    if not ok or not self.is_link_alive(timeout=2.5):
                        self.abort_transmission(f"Optical link interrupted during chunk {idx + 1}/{tot}")
                        return

                    with self.lock:
                        self.chunks_sent += 1

                    # Wait for optical confirmation of chunk reception
                    # (At 30 FPS, client renders in ~2ms, camera captures and server decodes in 40-100ms)
                    # Allow up to 1.0s to absorb temporary compositor/framing jitter (e.g. over RDP)
                    t_chunk_wait = time.time() + 1.0
                    while time.time() < t_chunk_wait:
                        time.sleep(0.01)
                        if not self.is_link_alive(timeout=2.5):
                            self.abort_transmission(f"Optical link lost while awaiting ACK for chunk {idx + 1}/{tot}")
                            return
                        if self.is_chunk_acked(idx):
                            chunk_acked = True
                            break

                    if chunk_acked:
                        break
                    elif attempt < max_chunk_retries - 1:
                        with self.lock:
                            self.retransmissions += 1
                        self.log(f"Chunk {idx + 1}/{tot} not confirmed within 1.0s, retrying (attempt {attempt + 2}/{max_chunk_retries})...")

                if not chunk_acked:
                    self.abort_transmission(
                        f"Chunk {idx + 1}/{tot} failed after {max_chunk_retries} attempts (optical ACK not received). "
                        "Halting to avoid typing into wrong window."
                    )
                    return

                time.sleep(self.packet_delay)

            # 3. Final verification of all chunks
            with self.lock:
                missing = [i for i in range(tot) if not staged["acked_chunks"][i]]

            if missing:
                self.log(f"Final pass: {len(missing)} unacked chunks remaining, retransmitting...")
                for idx in missing:
                    if not self.is_link_alive(timeout=2.5):
                        self.abort_transmission("Optical link lost during final verification")
                        return
                    pkt = staged["data_pkts"][idx]
                    ok = self.typer.type_string(pkt, delay=self.char_delay, cancel_check=cancel_data_cb)
                    if not ok or not self.is_link_alive(timeout=2.5):
                        return
                    t_chunk_wait = time.time() + 0.80
                    while time.time() < t_chunk_wait:
                        time.sleep(0.01)
                        if self.is_chunk_acked(idx):
                            break

            # 4. Send End of transmission packet
            if not self.is_link_alive(timeout=2.5):
                self.abort_transmission("Optical link lost before sending end packet")
                return

            self.log("All chunks transmitted & optically verified! Sending End-of-Stream packet...")
            for _ in range(3):
                if not self.is_link_alive(timeout=2.5):
                    self.abort_transmission("Optical link lost during end-of-stream")
                    return
                ok = self.typer.type_string(staged["end_pkt"], delay=self.char_delay, cancel_check=cancel_data_cb)
                if not ok or not self.is_link_alive(timeout=2.5):
                    return
                t_end = time.time() + 1.2
                while time.time() < t_end:
                    time.sleep(0.03)
                    with self.lock:
                        if self.client_state == STATE_COMPLETE:
                            self.transfer_active = False
                            self.transfer_complete = True
                            self.end_time = time.time()
                            self.log("Transmission 100% COMPLETE & VERIFIED by client optical grid!")
                            return
        finally:
            self.typer.release_all_keys()

    def send_chat_message(self, text):
        if not text:
            return
        with self.lock:
            if not self.optical_locked or self.client_state < STATE_FOCUSED:
                self.log("Cannot send keystrokes: client window is not optically locked and focused.")
                return
        pkt = make_chat_packet(text)
        self.log(f"Typing chat message over keyboard: '{text}'")
        self.typer.type_string(pkt, delay=self.char_delay)

    def reset_state(self):
        with self.lock:
            self.staged_file = None
            self.transfer_active = False
            self.transfer_complete = False
            self.transfer_error = None
            self.abort_requested = False
            self.chunks_sent = 0
            self.chunks_acked = 0
            self.focus_confirmed_event.clear()
        self.typer.release_all_keys()
        self.log("Transfer state reset.")


# ==============================================================================
# EMBEDDED WEB APPLICATION (HTML / JS / CSS)
# ==============================================================================

SERVER_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>otd Duplex | Full-Duplex Air-Gap Data Transfer</title>
<style>
  :root {
    --bg-base: #0a0e17;
    --bg-card: #121a2a;
    --bg-card-sub: #1a2438;
    --border: #233149;
    --border-light: #2d3f5d;
    --text-main: #f1f5f9;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --primary: #10b981;
    --primary-glow: rgba(16, 185, 129, 0.25);
    --accent-cyan: #06b6d4;
    --accent-blue: #3b82f6;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg-base);
    color: var(--text-main);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand-logo {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--primary), var(--accent-cyan));
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; color: #051410; font-size: 16px;
  }
  .brand-title {
    font-size: 18px; font-weight: 700; display: flex; align-items: center; gap: 8px;
  }
  .badge {
    font-size: 11px; padding: 2px 7px; border-radius: 4px; font-weight: 700;
  }
  .badge-cyan { background: var(--accent-cyan); color: #032128; }
  .badge-green { background: rgba(16, 185, 129, 0.15); color: var(--primary); border: 1px solid rgba(16, 185, 129, 0.3); }

  .header-actions { display: flex; align-items: center; gap: 12px; }
  .btn {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 8px 15px; border-radius: 8px; font-size: 13px; font-weight: 600;
    cursor: pointer; border: none; transition: all 0.15s ease;
  }
  .btn-primary { background: var(--primary); color: #042116; }
  .btn-danger { background: var(--accent-rose); color: #fff; }
  .btn-danger:hover { background: #e11d48; }
  @keyframes pulse-stop {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.7); }
    70% { transform: scale(1.02); box-shadow: 0 0 0 8px rgba(244, 63, 94, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(244, 63, 94, 0); }
  }
  .btn-danger-pulse {
    animation: pulse-stop 1.6s infinite;
  }
  .btn-outline { background: var(--bg-card-sub); color: var(--text-main); border: 1px solid var(--border); }
  .btn-outline:hover { background: var(--border); }

  /* Workflow Steps Banner */
  .workflow-strip {
    background: var(--bg-card-sub);
    border-bottom: 1px solid var(--border);
    padding: 10px 24px;
    display: flex;
    align-items: center;
    justify-content: space-around;
    font-size: 12px;
  }
  .step-item {
    display: flex; align-items: center; gap: 8px; color: var(--text-muted);
  }
  .step-item.active { color: var(--accent-cyan); font-weight: 700; }
  .step-item.done { color: var(--primary); font-weight: 600; }
  .step-num {
    width: 20px; height: 20px; border-radius: 50%;
    background: var(--border); color: var(--text-dim);
    display: flex; align-items: center; justify-content: center; font-size: 11px;
  }
  .step-item.active .step-num { background: var(--accent-cyan); color: #042128; }
  .step-item.done .step-num { background: var(--primary); color: #042128; }

  /* Main Layout */
  .main-container {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 20px;
    padding: 20px 24px;
    flex: 1;
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .card-header {
    padding: 12px 18px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .card-title {
    font-size: 14px; font-weight: 700; display: flex; align-items: center; gap: 8px;
  }
  .card-body { padding: 18px; display: flex; flex-direction: column; gap: 16px; flex: 1; }

  /* Video Stage & Masking */
  .mask-toolbar {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  }
  .video-stage {
    position: relative;
    background: #000;
    border-radius: 8px;
    overflow: hidden;
    aspect-ratio: 16 / 10;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  #videoEl { width: 100%; height: 100%; object-fit: contain; display: none; }
  #overlayCanvas {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;
  }
  .video-stage.draw-mode { cursor: crosshair; }
  .video-stage.draw-mode #overlayCanvas { pointer-events: auto; }
  .empty-stage {
    display: flex; flex-direction: column; align-items: center; gap: 10px;
    color: var(--text-dim); text-align: center; padding: 20px;
  }

  /* File Dropzone */
  .dropzone {
    border: 2px dashed var(--border-light);
    border-radius: 10px;
    padding: 22px;
    text-align: center;
    background: rgba(26, 36, 56, 0.4);
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: var(--accent-cyan);
    background: rgba(6, 182, 212, 0.08);
  }

  /* Focus Alert Card */
  .focus-alert {
    border-radius: 8px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    border: 1px solid var(--border);
  }
  .focus-waiting {
    background: rgba(245, 158, 11, 0.12);
    border-color: rgba(245, 158, 11, 0.4);
    color: var(--accent-amber);
  }
  .focus-confirmed {
    background: rgba(16, 185, 129, 0.12);
    border-color: rgba(16, 185, 129, 0.4);
    color: var(--primary);
  }

  /* Progress & Metrics */
  .metrics-grid {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;
  }
  .metric-card {
    background: var(--bg-card-sub);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
  }
  .metric-label { font-size: 11px; text-transform: uppercase; color: var(--text-dim); font-weight: 600; }
  .metric-val { font-size: 18px; font-weight: 700; color: var(--text-main); margin-top: 3px; }

  .progress-bar-bg {
    width: 100%; height: 10px; background: var(--bg-card-sub);
    border-radius: 999px; overflow: hidden; border: 1px solid var(--border);
  }
  .progress-bar-fill {
    height: 100%; width: 0%; background: linear-gradient(90deg, var(--primary), var(--accent-cyan));
    transition: width 0.1s linear;
  }

  /* Chat Console */
  .chat-box {
    display: flex; gap: 8px;
  }
  .chat-input {
    flex: 1; background: var(--bg-card-sub); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 12px; color: var(--text-main); font-size: 13px;
  }
  .chat-input:focus { outline: none; border-color: var(--accent-cyan); }

  /* Activity Log */
  .log-box {
    background: #080c14; border: 1px solid var(--border); border-radius: 8px;
    padding: 10px; font-family: monospace; font-size: 12px; height: 130px;
    overflow-y: auto; color: var(--text-muted);
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="brand-logo">2W</div>
    <div class="brand-title">
      otd Duplex
      <span class="badge badge-cyan">v2 Two-Way</span>
      <span class="badge badge-green" id="badgeTyper">Keystroke Sender: Ready</span>
    </div>
  </div>
  <div class="header-actions">
    <div style="display:flex; align-items:center; gap:6px;">
      <span style="font-size:11px; font-weight:700; color:var(--text-muted);">SPEED:</span>
      <select id="speedSelect" onchange="changeSpeed(this.value)" style="background:var(--bg-card-sub); color:var(--text-main); border:1px solid var(--border); border-radius:6px; padding:6px 8px; font-size:11px; font-weight:600; cursor:pointer;">
        <option value="turbo">⚡ Turbo (30-50 KB/s)</option>
        <option value="fast" selected>🚀 Fast (15-25 KB/s)</option>
        <option value="balanced">⚖ Balanced (5-10 KB/s)</option>
        <option value="safe">🛡️ Safe / VDI (1-3 KB/s)</option>
      </select>
    </div>
    <button id="btnAbort" class="btn btn-danger btn-danger-pulse" style="display:none;" onclick="abortTransfer()">
      🛑 Stop Transmission
    </button>
    <button id="btnRequestKb" class="btn btn-outline" onclick="requestKeyboardAccess()">
      🔑 Request Keyboard Access
    </button>
    <button id="btnShare" class="btn btn-primary" onclick="toggleScreenShare()">
      📺 Share Screen / Window
    </button>
    <button id="btnReset" class="btn btn-outline" onclick="resetTransfer()">
      ↺ Reset Transfer
    </button>
  </div>
</header>

<div class="workflow-strip">
  <div class="step-item active" id="step1">
    <div class="step-num">1</div>
    <span>Share Client Window & Set Mask</span>
  </div>
  <div class="step-item" id="step2">
    <div class="step-num">2</div>
    <span>Select File to Send</span>
  </div>
  <div class="step-item" id="step3">
    <div class="step-num">3</div>
    <span>Focus Client & Press Key</span>
  </div>
  <div class="step-item" id="step4">
    <div class="step-num">4</div>
    <span>Keyboard Transmission</span>
  </div>
  <div class="step-item" id="step5">
    <div class="step-num">5</div>
    <span>Optical Grid Verified</span>
  </div>
</div>

<div class="main-container">
  <!-- Left Column: Optical Feedback Video & Masking -->
  <div class="card">
    <div class="card-header">
      <div class="card-title">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
        Client Optical Feedback Stream
      </div>
      <div id="signalBadge" style="font-size:12px; font-weight:700; color:var(--accent-amber);">
        Waiting for Screen Share
      </div>
    </div>
    <div class="card-body">
      <div class="mask-toolbar">
        <button id="btnDrawMask" class="btn btn-outline" onclick="toggleDrawMask()">
          ✏ Draw Mask
        </button>
        <button id="btnAutoSnap" class="btn btn-outline" onclick="autoSnapMask()">
          🎯 Auto-Snap Mask
        </button>
        <button id="btnClearMask" class="btn btn-outline" style="display:none;" onclick="clearMask()">
          ✖ Clear Mask
        </button>
        <div id="maskLabel" style="font-size:12px; color:var(--text-muted); margin-left:auto;">Full Window (No Mask)</div>
      </div>

      <div id="videoStage" class="video-stage">
        <video id="videoEl" autoplay playsinline muted></video>
        <canvas id="overlayCanvas"></canvas>
        <div id="emptyStage" class="empty-stage">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
          <div style="font-size:15px; font-weight:600; color:var(--text-main);">No Client Video Feed</div>
          <div style="font-size:13px; max-width:340px;">Click "Share Screen / Window" above and select the client.py window.</div>
        </div>
      </div>

      <!-- Chat / Keystroke Sender Box -->
      <div>
        <div style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-dim); margin-bottom:6px;">Chat / Keystroke Command Console</div>
        <div class="chat-box">
          <input type="text" id="chatInput" class="chat-input" placeholder="Type a message or command to type into client..." onkeydown="if(event.key==='Enter') sendChatMessage()">
          <button class="btn btn-outline" onclick="sendChatMessage()">Send Keystrokes</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Right Column: File Sender, Telemetry & Focus Confirmation -->
  <div class="card">
    <div class="card-header">
      <div class="card-title">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        File Staging & Keystroke Transmission
      </div>
      <div id="stageBadge" style="font-size:12px; font-weight:700; color:var(--text-muted);">Idle</div>
    </div>
    <div class="card-body">
      <!-- File Selector Dropzone -->
      <div id="dropzone" class="dropzone" onclick="document.getElementById('fileInput').click()">
        <input type="file" id="fileInput" style="display:none;" onchange="handleFileSelected(event)">
        <div style="font-weight:600; font-size:14px; margin-bottom:4px;" id="dropzoneTitle">Select or Drag & Drop File to Send</div>
        <div style="font-size:12px; color:var(--text-muted);" id="dropzoneSub">Any file format (.zip, .tar, .bin, .pdf, .txt, scripts)</div>
      </div>

      <!-- Focus State Alert Banner -->
      <div id="focusAlert" class="focus-alert focus-waiting">
        <div style="font-size:20px;">⌨</div>
        <div>
          <div style="font-weight:700; font-size:13px;" id="focusTitle">Step 3: Awaiting Client Focus</div>
          <div style="font-size:12px; color:var(--text-muted);" id="focusDesc">
            After staging a file, focus the client terminal (or GUI window) and press ENTER or SPACE.
          </div>
        </div>
      </div>

      <!-- Progress Bar -->
      <div>
        <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:600; margin-bottom:6px;">
          <span style="color:var(--text-muted); text-transform:uppercase;">Transmission Progress</span>
          <span id="pctVal" style="color:var(--accent-cyan);">0%</span>
        </div>
        <div class="progress-bar-bg">
          <div id="progressBar" class="progress-bar-fill"></div>
        </div>
      </div>

      <!-- Telemetry Cards -->
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-label">Chunks ACKed (Grid)</div>
          <div id="chunksVal" class="metric-val">0 / 0</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Typing Throughput</div>
          <div id="speedVal" class="metric-val">0.0 B/s</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Client Optical State</div>
          <div id="stateVal" class="metric-val" style="font-size:14px; color:var(--accent-amber);">WAIT_FOCUS</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Retransmissions</div>
          <div id="retransVal" class="metric-val">0</div>
        </div>
      </div>

      <!-- Event Activity Log -->
      <div>
        <div style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-dim); margin-bottom:6px;">Duplex Activity Log</div>
        <div id="logBox" class="log-box"></div>
      </div>
    </div>
  </div>
</div>

<script>
  let captureStream = null;
  let isCapturing = false;
  let cropRegion = null;
  let isDrawingMask = false;
  let drawStartX = 0, drawStartY = 0;
  let lastFiducials = null;
  let ws = null;
  let captureInterval = null;

  const videoEl = document.getElementById('videoEl');
  const overlayCanvas = document.getElementById('overlayCanvas');
  const emptyStage = document.getElementById('emptyStage');
  const videoStage = document.getElementById('videoStage');
  const logBox = document.getElementById('logBox');

  function logMsg(msg) {
    const ts = new Date().toTimeString().split(' ')[0];
    const div = document.createElement('div');
    div.textContent = `[${ts}] ${msg}`;
    logBox.appendChild(div);
    logBox.scrollTop = logBox.scrollHeight;
  }

  // WebSocket Connection
  function initWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      logMsg('Connected to Server Duplex WebSocket.');
    };

    ws.onmessage = (evt) => {
      isSending = false;
      if (sendWatchdog) {
        clearTimeout(sendWatchdog);
        sendWatchdog = null;
      }
      try {
        const data = JSON.parse(evt.data);
        updateTelemetryUI(data);
      } catch (e) {}
    };

    ws.onclose = () => {
      isSending = false;
      if (sendWatchdog) {
        clearTimeout(sendWatchdog);
        sendWatchdog = null;
      }
      setTimeout(initWebSocket, 2000);
    };
    ws.onerror = () => {
      isSending = false;
      if (sendWatchdog) {
        clearTimeout(sendWatchdog);
        sendWatchdog = null;
      }
    };
  }
  initWebSocket();

  // Screen Sharing
  async function toggleScreenShare() {
    if (isCapturing) {
      stopCapture();
    } else {
      await startCapture();
    }
  }

  async function startCapture() {
    try {
      captureStream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: "window", frameRate: { ideal: 30, max: 60 } },
        audio: false
      });
      videoEl.srcObject = captureStream;
      videoEl.style.display = 'block';
      overlayCanvas.style.display = 'block';
      emptyStage.style.display = 'none';
      await videoEl.play();

      isCapturing = true;
      document.getElementById('btnShare').textContent = '⏹ Stop Sharing';
      document.getElementById('btnShare').classList.replace('btn-primary', 'btn-danger');
      document.getElementById('signalBadge').textContent = 'Searching for Client Grid';
      document.getElementById('signalBadge').style.color = 'var(--accent-amber)';

      captureStream.getVideoTracks()[0].onended = stopCapture;
      startFrameStreaming();
      setWorkflowStep(1, true);
      logMsg('Screen capture started. Looking for client corner fiducials...');
    } catch (e) {
      logMsg(`Screen share cancelled or failed: ${e}`);
    }
  }

  function stopCapture() {
    // Immediately abort transmission on server if screen share is stopped
    fetch('/api/abort', { method: 'POST' }).catch(() => {});
    stopFrameStreaming();
    if (captureInterval) clearInterval(captureInterval);
    if (captureStream) captureStream.getTracks().forEach(t => t.stop());
    videoEl.style.display = 'none';
    overlayCanvas.style.display = 'none';
    emptyStage.style.display = 'flex';
    isCapturing = false;
    document.getElementById('btnShare').textContent = '📺 Share Screen / Window';
    document.getElementById('btnShare').classList.replace('btn-danger', 'btn-primary');
    document.getElementById('signalBadge').textContent = 'Waiting for Screen Share';
    document.getElementById('signalBadge').style.color = 'var(--accent-amber)';
    const btnAbort = document.getElementById('btnAbort');
    if (btnAbort) btnAbort.style.display = 'none';
  }

  function abortTransfer() {
    logMsg('Emergency stop triggered! Stopping keystroke transmission immediately...');
    fetch('/api/abort', { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        logMsg('Transmission halted. Keystroke sender stopped.');
        const btnAbort = document.getElementById('btnAbort');
        if (btnAbort) btnAbort.style.display = 'none';
      });
  }

  function changeSpeed(preset) {
    logMsg(`Updating speed preset to: ${preset.toUpperCase()}...`);
    fetch('/api/speed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset: preset })
    })
    .then(r => r.json())
    .then(d => {
      logMsg(`Speed updated: ${d.preset.toUpperCase()} (${d.chunk_size}B chunks, ${d.char_delay * 1000}ms delay)`);
    });
  }

  // Video Frame Streaming & Masking
  const offCanvas = document.createElement('canvas');
  const offCtx = offCanvas.getContext('2d');
  let isSending = false;
  let sendWatchdog = null;
  let isDragging = false;
  let dragStart = null;
  let dragCurrent = null;

  function requestKeyboardAccess() {
    logMsg('Requesting OS keyboard / remote control access...');
    fetch('/api/request_keyboard', { method: 'POST' })
      .then(r => r.json())
      .then(data => {
        logMsg(data.message);
        const badge = document.getElementById('badgeTyper');
        badge.textContent = `Keystroke Sender: ${data.backend.toUpperCase()} (Active)`;
        badge.className = 'badge badge-green';
      });
  }

  function getVideoDisplayRect() {
    const stageW = videoStage.clientWidth;
    const stageH = videoStage.clientHeight;
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    if (!vw || !vh) return null;

    const stageAspect = stageW / stageH;
    const videoAspect = vw / vh;

    let dw, dh, dx, dy;
    if (stageAspect > videoAspect) {
      dh = stageH;
      dw = stageH * videoAspect;
      dx = (stageW - dw) / 2;
      dy = 0;
    } else {
      dw = stageW;
      dh = stageW / videoAspect;
      dx = 0;
      dy = (stageH - dh) / 2;
    }
    return { dx, dy, dw, dh, videoW: vw, videoH: vh, stageW, stageH };
  }

  function getPointerVideoCoords(e) {
    const rect = getVideoDisplayRect();
    if (!rect) return null;
    const canvasRect = overlayCanvas.getBoundingClientRect();
    const pointerX = e.clientX - canvasRect.left;
    const pointerY = e.clientY - canvasRect.top;

    const clampedX = Math.max(rect.dx, Math.min(rect.dx + rect.dw, pointerX));
    const clampedY = Math.max(rect.dy, Math.min(rect.dy + rect.dh, pointerY));

    const vx = ((clampedX - rect.dx) / rect.dw) * rect.videoW;
    const vy = ((clampedY - rect.dy) / rect.dh) * rect.videoH;
    return { x: vx, y: vy };
  }

  // Background-safe unthrottled frame capture worker (prevents browser tab throttling)
  let timerWorker = null;
  try {
    const workerBlob = new Blob([`
      let timer = null;
      self.onmessage = function(e) {
        if (e.data === 'start') {
          if (!timer) timer = setInterval(function() { self.postMessage('tick'); }, 25);
        } else if (e.data === 'stop') {
          if (timer) { clearInterval(timer); timer = null; }
        }
      };
    `], { type: 'application/javascript' });
    timerWorker = new Worker(URL.createObjectURL(workerBlob));
  } catch (e) {}

  function sendNextFrame() {
    if (!isCapturing) return;
    if (!isSending && ws && ws.readyState === WebSocket.OPEN && videoEl.videoWidth > 0) {
      isSending = true;
      clearTimeout(sendWatchdog);
      sendWatchdog = setTimeout(() => { isSending = false; }, 800);
      try {
        const vw = videoEl.videoWidth;
        const vh = videoEl.videoHeight;
        if (cropRegion && cropRegion.w > 40 && cropRegion.h > 40) {
          offCanvas.width = cropRegion.w;
          offCanvas.height = cropRegion.h;
          offCtx.drawImage(videoEl, cropRegion.x, cropRegion.y, cropRegion.w, cropRegion.h, 0, 0, cropRegion.w, cropRegion.h);
        } else {
          const maxDim = Math.max(vw, vh);
          const scale = maxDim > 1280 ? (1280 / maxDim) : 1;
          offCanvas.width = Math.round(vw * scale);
          offCanvas.height = Math.round(vh * scale);
          offCtx.drawImage(videoEl, 0, 0, offCanvas.width, offCanvas.height);
        }

        offCanvas.toBlob((blob) => {
          if (blob && ws && ws.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then(buf => {
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(buf);
              } else {
                isSending = false;
              }
            }).catch(() => {
              isSending = false;
            });
          } else {
            isSending = false;
            if (sendWatchdog) clearTimeout(sendWatchdog);
          }
        }, 'image/jpeg', 0.85);
      } catch (e) {
        isSending = false;
        if (sendWatchdog) clearTimeout(sendWatchdog);
      }
    }
  }

  function startFrameStreaming() {
    if (timerWorker) {
      timerWorker.onmessage = () => sendNextFrame();
      timerWorker.postMessage('start');
    }
    // Also use requestVideoFrameCallback for direct hardware-synchronized frames
    function onVideoFrame() {
      if (!isCapturing) return;
      sendNextFrame();
      if ('requestVideoFrameCallback' in videoEl) {
        videoEl.requestVideoFrameCallback(onVideoFrame);
      }
    }
    if ('requestVideoFrameCallback' in videoEl) {
      videoEl.requestVideoFrameCallback(onVideoFrame);
    }
  }

  function stopFrameStreaming() {
    if (timerWorker) timerWorker.postMessage('stop');
  }

  // Masking
  function toggleDrawMask() {
    isDrawingMask = !isDrawingMask;
    const btn = document.getElementById('btnDrawMask');
    if (isDrawingMask) {
      btn.classList.add('active');
      btn.textContent = 'Drawing...';
      videoStage.classList.add('draw-mode');
      logMsg('Click and drag across client transmission area to set mask.');
    } else {
      btn.classList.remove('active');
      btn.textContent = '✏ Draw Mask';
      videoStage.classList.remove('draw-mode');
    }
  }

  overlayCanvas.addEventListener('pointerdown', (e) => {
    if (!isDrawingMask || !isCapturing || videoEl.videoWidth === 0) return;
    const pt = getPointerVideoCoords(e);
    if (!pt) return;
    isDragging = true;
    dragStart = pt;
    dragCurrent = pt;
    overlayCanvas.setPointerCapture(e.pointerId);
  });

  window.addEventListener('pointermove', (e) => {
    if (!isDragging) return;
    const pt = getPointerVideoCoords(e);
    if (!pt) return;
    dragCurrent = pt;
    redrawOverlay();
  });

  window.addEventListener('pointerup', (e) => {
    if (!isDragging) return;
    isDragging = false;
    if (dragStart && dragCurrent) {
      const x = Math.max(0, Math.min(dragStart.x, dragCurrent.x));
      const y = Math.max(0, Math.min(dragStart.y, dragCurrent.y));
      const w = Math.min(videoEl.videoWidth - x, Math.abs(dragCurrent.x - dragStart.x));
      const h = Math.min(videoEl.videoHeight - y, Math.abs(dragCurrent.y - dragStart.y));
      if (w > 40 && h > 40) {
        cropRegion = { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) };
        document.getElementById('btnClearMask').style.display = 'inline-flex';
        document.getElementById('maskLabel').textContent = `Mask: ${cropRegion.w}×${cropRegion.h} px`;
        logMsg(`Transmission mask set: ${cropRegion.w}×${cropRegion.h}`);
      }
    }
    dragStart = null;
    dragCurrent = null;
    toggleDrawMask();
    redrawOverlay();
  });

  function autoSnapMask() {
    if (!lastFiducials || lastFiducials.length !== 4) {
      logMsg('Auto-Snap: Waiting for optical fiducials to be detected first.');
      return;
    }
    const xs = lastFiducials.map(p => p[0]);
    const ys = lastFiducials.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const padX = (maxX - minX) * 0.12;
    const padY = (maxY - minY) * 0.12;

    const cx = Math.max(0, Math.round(minX - padX));
    const cy = Math.max(0, Math.round(minY - padY));
    const cw = Math.min(videoEl.videoWidth - cx, Math.round((maxX - minX) + padX * 2));
    const ch = Math.min(videoEl.videoHeight - cy, Math.round((maxY - minY) + padY * 2));

    cropRegion = { x: cx, y: cy, w: cw, h: ch };
    document.getElementById('btnClearMask').style.display = 'inline-flex';
    document.getElementById('maskLabel').textContent = `Auto-Snapped: ${cw}×${ch} px`;
    logMsg(`Auto-snapped mask around client fiducials: ${cw}×${ch}`);
    redrawOverlay();
  }

  function clearMask() {
    cropRegion = null;
    document.getElementById('btnClearMask').style.display = 'none';
    document.getElementById('maskLabel').textContent = 'Full Window (No Mask)';
    logMsg('Transmission mask cleared.');
    redrawOverlay();
  }

  function redrawOverlay(fiducials = lastFiducials) {
    const rect = getVideoDisplayRect();
    if (!rect) return;

    if (overlayCanvas.width !== rect.stageW || overlayCanvas.height !== rect.stageH) {
      overlayCanvas.width = rect.stageW;
      overlayCanvas.height = rect.stageH;
    }

    const ctx = overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    // 1. Draw transmission mask cutout & reticle
    if (cropRegion && cropRegion.w > 20 && cropRegion.h > 20) {
      const dx = rect.dx + (cropRegion.x / rect.videoW) * rect.dw;
      const dy = rect.dy + (cropRegion.y / rect.videoH) * rect.dh;
      const dw = (cropRegion.w / rect.videoW) * rect.dw;
      const dh = (cropRegion.h / rect.videoH) * rect.dh;

      ctx.save();
      ctx.fillStyle = 'rgba(10, 14, 23, 0.70)';
      ctx.beginPath();
      ctx.rect(0, 0, overlayCanvas.width, overlayCanvas.height);
      ctx.rect(dx, dy, dw, dh);
      ctx.fill('evenodd');

      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;
      ctx.strokeRect(dx, dy, dw, dh);

      const clen = Math.min(20, dw / 4, dh / 4);
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(dx, dy + clen); ctx.lineTo(dx, dy); ctx.lineTo(dx + clen, dy);
      ctx.moveTo(dx + dw - clen, dy); ctx.lineTo(dx + dw, dy); ctx.lineTo(dx + dw, dy + clen);
      ctx.moveTo(dx + dw, dy + dh - clen); ctx.lineTo(dx + dw, dy + dh); ctx.lineTo(dx + dw - clen, dy + dh);
      ctx.moveTo(dx + clen, dy + dh); ctx.lineTo(dx, dy + dh); ctx.lineTo(dx, dy + dh - clen);
      ctx.stroke();

      ctx.fillStyle = '#06b6d4';
      ctx.font = 'bold 11px monospace';
      ctx.fillText(`MASK (${cropRegion.w}×${cropRegion.h} px)`, dx + 4, Math.max(14, dy - 5));
      ctx.restore();
    }

    // 2. Draw active dragging selection
    if (isDragging && dragStart && dragCurrent) {
      const x1 = Math.min(dragStart.x, dragCurrent.x);
      const y1 = Math.min(dragStart.y, dragCurrent.y);
      const w = Math.abs(dragCurrent.x - dragStart.x);
      const h = Math.abs(dragCurrent.y - dragStart.y);

      const dx = rect.dx + (x1 / rect.videoW) * rect.dw;
      const dy = rect.dy + (y1 / rect.videoH) * rect.dh;
      const dw = (w / rect.videoW) * rect.dw;
      const dh = (h / rect.videoH) * rect.dh;

      ctx.save();
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
      ctx.fillRect(dx, dy, dw, dh);
      ctx.strokeRect(dx, dy, dw, dh);
      ctx.restore();
    }

    // 3. Draw detected optical fiducials (mapped to display coordinates)
    if (fiducials && fiducials.length === 4) {
      const toDisplayPt = (pt) => [
        rect.dx + (pt[0] / rect.videoW) * rect.dw,
        rect.dy + (pt[1] / rect.videoH) * rect.dh
      ];
      const [tl, tr, br, bl] = fiducials.map(toDisplayPt);

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(tl[0], tl[1]);
      ctx.lineTo(tr[0], tr[1]);
      ctx.lineTo(br[0], br[1]);
      ctx.lineTo(bl[0], bl[1]);
      ctx.closePath();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#10b981';
      ctx.stroke();

      const corners = [
        { pt: tl, col: '#ef4444' }, // TL Red
        { pt: tr, col: '#22c55e' }, // TR Green
        { pt: bl, col: '#3b82f6' }, // BL Blue
        { pt: br, col: '#d946ef' }  // BR Magenta
      ];
      corners.forEach(c => {
        ctx.beginPath();
        ctx.arc(c.pt[0], c.pt[1], 5, 0, 2 * Math.PI);
        ctx.fillStyle = c.col;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
      });
      ctx.restore();
    }
  }

  // Telemetry Updates
  function updateTelemetryUI(data) {
    if (data.optical_locked) {
      document.getElementById('signalBadge').textContent = '● Optical Signal Locked';
      document.getElementById('signalBadge').style.color = 'var(--primary)';
      setWorkflowStep(1, true);
    }
    if (data.fiducials && data.fiducials.length === 4) {
      let adj = data.fiducials;
      if (cropRegion && cropRegion.w > 40) {
        adj = data.fiducials.map(p => [p[0] + cropRegion.x, p[1] + cropRegion.y]);
      }
      lastFiducials = adj;
    }
    redrawOverlay(lastFiducials);

    document.getElementById('stateVal').textContent = data.client_state;
    document.getElementById('chunksVal').textContent = `${data.chunks_acked} / ${data.total_chunks}`;
    document.getElementById('speedVal').textContent = `${data.speed_bps} B/s`;
    document.getElementById('pctVal').textContent = `${data.progress_pct}%`;
    document.getElementById('progressBar').style.width = `${data.progress_pct}%`;
    document.getElementById('retransVal').textContent = data.retransmissions;

    const focusAlert = document.getElementById('focusAlert');
    const focusTitle = document.getElementById('focusTitle');
    const focusDesc = document.getElementById('focusDesc');

    if (data.focus_confirmed || data.client_state_code >= 1) {
      focusAlert.className = 'focus-alert focus-confirmed';
      focusTitle.textContent = 'Step 3: Client Keyboard Focus Confirmed!';
      focusDesc.textContent = 'Client has signaled focus via optical grid. Ready for transmission.';
      setWorkflowStep(3, true);
    } else {
      focusAlert.className = 'focus-alert focus-waiting';
      focusTitle.textContent = 'Step 3: Awaiting Client Focus';
      focusDesc.textContent = 'Focus the client terminal (or GUI window) and press ENTER or SPACE.';
    }

    const btnAbort = document.getElementById('btnAbort');
    if (btnAbort) {
      btnAbort.style.display = data.transfer_active ? 'inline-flex' : 'none';
    }

    if (data.speed_preset) {
      const sp = document.getElementById('speedSelect');
      if (sp && sp.value !== data.speed_preset) sp.value = data.speed_preset;
    }

    if (data.transfer_active) {
      setWorkflowStep(4, true);
      document.getElementById('stageBadge').textContent = 'Transmitting...';
      document.getElementById('stageBadge').style.color = 'var(--accent-cyan)';
    } else if (data.abort_requested) {
      document.getElementById('stageBadge').textContent = 'Transmission Stopped';
      document.getElementById('stageBadge').style.color = 'var(--accent-rose)';
    }

    if (data.transfer_complete) {
      setWorkflowStep(5, true);
      document.getElementById('stageBadge').textContent = 'Verified & Complete!';
      document.getElementById('stageBadge').style.color = 'var(--primary)';
    }
  }

  function setWorkflowStep(stepNum, isDone) {
    for (let i = 1; i <= 5; i++) {
      const el = document.getElementById(`step${i}`);
      if (i < stepNum) {
        el.className = 'step-item done';
      } else if (i === stepNum) {
        el.className = isDone ? 'step-item done' : 'step-item active';
      } else {
        el.className = 'step-item';
      }
    }
  }

  // File Staging (Upload to Server)
  function handleFileSelected(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    logMsg(`Staging file '${file.name}' (${file.size} bytes)...`);
    fetch('/api/upload', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(resp => {
        if (resp.status === 'ok') {
          document.getElementById('dropzoneTitle').textContent = `Staged: ${file.name}`;
          document.getElementById('dropzoneSub').textContent = `${(file.size / 1024).toFixed(1)} KB | Ready to transmit`;
          setWorkflowStep(2, true);
          logMsg(`File staged. Please focus client window and press Enter/Space!`);
        } else {
          logMsg(`Upload error: ${resp.message}`);
        }
      });
  }

  function resetTransfer() {
    fetch('/api/reset', { method: 'POST' }).then(() => {
      document.getElementById('dropzoneTitle').textContent = 'Select or Drag & Drop File to Send';
      document.getElementById('dropzoneSub').textContent = 'Any file format (.zip, .tar, .bin, .pdf, .txt, scripts)';
      document.getElementById('progressBar').style.width = '0%';
      document.getElementById('pctVal').textContent = '0%';
      setWorkflowStep(1, false);
      logMsg('Transfer reset.');
    });
  }

  function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if (!msg) return;

    fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg })
    }).then(r => r.json()).then(resp => {
      logMsg(`Sent message keystrokes: "${msg}"`);
      input.value = '';
    });
  }
</script>
</body>
</html>
"""


# ==============================================================================
# HTTP & WEBSOCKET SERVER
# ==============================================================================

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_duplex_http_handler(session):
    class DuplexHTTPHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/ws" and self.headers.get("Upgrade", "").lower() == "websocket":
                self.handle_websocket()
            elif self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(SERVER_HTML_PAGE.encode("utf-8"))
            elif self.path == "/api/status":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                status_dict = {
                    "optical_locked": session.optical_locked,
                    "client_state": STATE_NAMES.get(session.client_state, "UNKNOWN"),
                    "focus_confirmed": session.focus_confirmed_event.is_set(),
                    "transfer_active": session.transfer_active,
                    "transfer_complete": session.transfer_complete,
                    "chunks_sent": session.chunks_sent,
                    "chunks_acked": session.chunks_acked
                }
                self.wfile.write(json.dumps(status_dict).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/api/upload":
                # Parse multipart or raw file upload
                ctype = self.headers.get("Content-Type", "")
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)

                filename = "transfer_file.bin"
                raw_bytes = body
                if "multipart/form-data" in ctype:
                    # Simple multipart extraction
                    boundary = ctype.split("boundary=")[-1].encode("ascii")
                    parts = body.split(b"--" + boundary)
                    for part in parts:
                        if b'Content-Disposition: form-data;' in part and b'filename="' in part:
                            header_part, file_part = part.split(b"\r\n\r\n", 1)
                            for line in header_part.decode("utf-8", errors="replace").split("\r\n"):
                                if "filename=" in line:
                                    filename = line.split('filename="')[1].split('"')[0]
                            raw_bytes = file_part.rstrip(b"\r\n")
                            break

                ok, msg = session.stage_file(filename, raw_bytes)
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok" if ok else "error", "message": msg}).encode("utf-8"))

            elif self.path == "/api/chat":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body.decode("utf-8"))
                    msg = data.get("message", "")
                    session.send_chat_message(msg)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
                except Exception as e:
                    self.send_response(400)
                    self.end_headers()

            elif self.path == "/api/reset":
                session.reset_state()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))

            elif self.path == "/api/abort":
                session.abort_transmission("User or browser requested abort")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "message": "Transmission halted"}).encode("utf-8"))

            elif self.path == "/api/speed":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body.decode("utf-8"))
                    preset = data.get("preset", "fast")
                    session.set_speed_preset(preset)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "ok",
                        "preset": preset,
                        "chunk_size": session.chunk_size,
                        "char_delay": session.char_delay
                    }).encode("utf-8"))
                except Exception:
                    self.send_response(400)
                    self.end_headers()

            elif self.path == "/api/request_keyboard":
                ok = session.typer.request_access()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "ok",
                    "backend": session.typer.backend,
                    "requested": ok,
                    "message": "Harmless keystroke simulated. If GNOME displayed a system prompt ('Allow remote control'), please click 'Allow'."
                }).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def handle_websocket(self):
            key = self.headers.get("Sec-WebSocket-Key", "")
            if not key:
                self.send_error(400, "Missing Sec-WebSocket-Key")
                return

            guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
            accept_val = base64.b64encode(hashlib.sha1((key.strip() + guid).encode("utf-8")).digest()).decode("utf-8")
            handshake = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_val}\r\n\r\n"
            )
            self.wfile.write(handshake.encode("utf-8"))
            self.wfile.flush()

            sock = self.connection
            sock.settimeout(15.0)

            def recv_exact(n):
                buf = bytearray()
                while len(buf) < n:
                    try:
                        c = sock.recv(n - len(buf))
                        if not c:
                            return None
                        buf.extend(c)
                    except Exception:
                        return None
                return bytes(buf)

            def send_ws_frame(opcode, payload_bytes):
                plen = len(payload_bytes)
                if plen < 126:
                    hdr = struct.pack(">BB", 0x80 | (opcode & 0x0f), plen)
                elif plen < 65536:
                    hdr = struct.pack(">BBH", 0x80 | (opcode & 0x0f), 126, plen)
                else:
                    hdr = struct.pack(">BBQ", 0x80 | (opcode & 0x0f), 127, plen)
                try:
                    self.wfile.write(hdr + payload_bytes)
                    self.wfile.flush()
                except Exception:
                    pass

            try:
                while True:
                    b12 = recv_exact(2)
                    if not b12:
                        break
                    opcode = b12[0] & 0x0f
                    masked = (b12[1] & 0x80) != 0
                    plen = b12[1] & 0x7f

                    if opcode == 0x8:  # Close
                        break
                    if plen == 126:
                        ext = recv_exact(2)
                        if not ext: break
                        plen = struct.unpack(">H", ext)[0]
                    elif plen == 127:
                        ext = recv_exact(8)
                        if not ext: break
                        plen = struct.unpack(">Q", ext)[0]

                    mask_key = recv_exact(4) if masked else None
                    raw = recv_exact(plen)
                    if raw is None or len(raw) < plen:
                        break

                    if masked and mask_key:
                        raw_arr = np.frombuffer(raw, dtype=np.uint8)
                        k_arr = np.frombuffer(mask_key, dtype=np.uint8)
                        payload = (raw_arr ^ np.resize(k_arr, len(raw_arr))).tobytes()
                    else:
                        payload = raw

                    if opcode == 0x2:  # Binary frame (JPEG video frame)
                        resp = session.process_image_bytes(payload)
                        send_ws_frame(0x1, json.dumps(resp).encode("utf-8"))
            except Exception:
                pass
            finally:
                session.abort_transmission("Screen share / WebSocket session ended")

        def log_message(self, format, *args):
            return

    return DuplexHTTPHandler


def run_server(port=8080, chunk_size=256, delay=0.0002, open_browser=True):
    session = ServerDuplexSession(chunk_size=chunk_size, char_delay=delay)

    server = None
    target_port = port
    for p in range(target_port, target_port + 20):
        try:
            handler_cls = make_duplex_http_handler(session)
            server = ThreadedHTTPServer(("0.0.0.0", p), handler_cls)
            target_port = p
            break
        except OSError:
            continue

    if server is None:
        print(f"Error: Unable to bind to port {port} or next 20 ports.")
        return False

    url = f"http://localhost:{target_port}"
    print("=" * 65)
    print("   OTD (OPTICAL DATA TRANSFER) DUPLEX SERVER (AIR-GAP SENDER)")
    print("=" * 65)
    print(f"Web Control Interface: {url}")
    print(f"Keystroke Synthesizer:  {session.typer.backend} (Alphanumeric only)")
    print(f"Packet Chunk Size:      {chunk_size} bytes / chunk")
    print(f"Character Typing Delay: {delay * 1000:.2f} ms")
    print(f"Dead-Man's Switch:      ACTIVE (halts typing if optical ACKs stall)")
    print("=" * 65)

    print("[Server] Keyboard synthesizer access: On-demand (via UI button or file staging)")
    print("Workflow:")
    print("  1. In the browser, click 'Share Screen/Window' and select client.py.")
    print("  2. Set or Auto-Snap the mask around the client optical grid.")
    print("  3. Drag & drop or select a file to send.")
    print("  4. Switch focus to client.py terminal (or window) and press ENTER or SPACE.")
    print("  5. Server will automatically detect focus via optical grid and type!")
    print("=" * 65)
    print("Press Ctrl+C to stop the server.\n")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Shutting down server...]")
    finally:
        server.shutdown()
        server.server_close()
        print("Server stopped.")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="otd Duplex Server (Air-Gap Sender & Optical Receiver)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for the browser web interface (default: 8080)")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Data payload bytes per keyboard packet (default: 256)")
    parser.add_argument("--delay", type=float, default=0.0002,
                        help="Typing delay per character in seconds (default: 0.0002s)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not launch browser automatically")
    args = parser.parse_args()

    run_server(port=args.port, chunk_size=args.chunk_size, delay=args.delay, open_browser=not args.no_browser)
