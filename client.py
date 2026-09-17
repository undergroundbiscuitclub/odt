#!/usr/bin/env python3
"""
client.py - Pure Python Standalone Duplex Client (Air-Gap Receiver & Optical ACK Sender)
Zero third-party dependencies (Python 3 standard library only).
Optimized for copy-pasting, locked-down RDP/VDI environments, minimal containers,
and air-gapped systems. Requires NO OpenCV and NO NumPy.

Channels:
- Visual / Optical (Client -> Server): Displays 4 corner chromatic fiducials and
  an optical ACK grid (structured ACK2 frame) via Tkinter GUI or ANSI terminal mode (--tty).
- Keyboard (Server -> Client): Reads incoming alphanumeric packets (hex framing + CRC32)
  via Tkinter window events and/or terminal stdin.

Workflow:
1. Start client.py (displays grid in WAITING_FOR_FOCUS state).
2. Share the client window or terminal in server.py's browser web UI and set the mask.
3. Select a file to send in the browser on server.py.
4. Click/focus the client window and press ENTER or SPACE.
5. Client signals focus via the optical grid; server automatically types the data!
6. Optical grid ACKs each received chunk and confirms end-to-end SHA-256 verification.
"""

import argparse
import hashlib
import os
import queue
import select
import struct
import sys
import threading
import time
import zlib

# 8 Saturated RGB Palette Colors (3 bits per cell)
# Format: (R, G, B)
PALETTE = [
    (0, 0, 0),       # 000: Black
    (0, 0, 255),     # 001: Blue
    (0, 255, 0),     # 010: Green
    (0, 255, 255),   # 011: Cyan
    (255, 0, 0),     # 100: Red
    (255, 0, 255),   # 101: Magenta
    (255, 255, 0),   # 110: Yellow
    (255, 255, 255)  # 111: White
]
PAL_BYTES = [bytes(c) for c in PALETTE]

# Chromatic fiducials: Red TL, Green TR, Blue BL, Magenta BR
FID_COLS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
FID_BYTES = [bytes(c) for c in FID_COLS]

# Precomputed ANSI blocks for headless terminal mode
ANSI_RGB_BLOCKS = [f"\033[48;2;{r};{g};{b}m  " for r, g, b in PALETTE]
ANSI_WHITE_BLOCK = "\033[48;2;255;255;255m  "
ANSI_BLACK_BLOCK = "\033[48;2;0;0;0m  "
ANSI_FIDS = [
    "\033[48;2;255;0;0m  ",    # TL Red
    "\033[48;2;0;255;0m  ",    # TR Green
    "\033[48;2;0;0;255m  ",    # BL Blue
    "\033[48;2;255;0;255m  "   # BR Magenta
]
ANSI_RESET = "\033[0m"

# Client lifecycle states
STATE_WAIT_FOCUS = 0    # Waiting for user to click & press Enter/Space
STATE_FOCUSED = 1       # Focus confirmed, waiting for server keyboard transmission
STATE_RECEIVING = 2     # Actively receiving file data chunks
STATE_COMPLETE = 3      # File fully reconstructed, decompressed & SHA-256 verified
STATE_ERROR = 4         # Checksum error or corrupted transfer

STATE_LABELS = {
    STATE_WAIT_FOCUS: "WAITING FOR FOCUS",
    STATE_FOCUSED: "KEYBOARD FOCUSED",
    STATE_RECEIVING: "RECEIVING DATA",
    STATE_COMPLETE: "VERIFIED & SAVED",
    STATE_ERROR: "TRANSFER ERROR"
}


# ==============================================================================
# PURE PYTHON OPTICAL FRAME ENCODING (Zero NumPy / OpenCV)
# ==============================================================================

def pack_ack_frame(state, last_ack, total_chunks, received_count, bytes_received,
                   bitmask_bytes, sha256_digest=b"", error_code=0, focus_seq=0, grid_size=32):
    """
    Constructs a valid Q2 visual frame containing structured optical ACK telemetry.
    Compatible with decoder_v2's FiducialTracker. Pure standard library only.
    Returns: list of int cell color indices [0..7], length = grid_size * grid_size.
    """
    bitmask_64 = bitmask_bytes.ljust(64, b"\x00")[:64]
    sha_32 = sha256_digest.ljust(32, b"\x00")[:32]
    err_byte = (error_code & 0x0F) | ((focus_seq & 0x0F) << 4)

    # Payload (118 bytes)
    body = struct.pack(">4sBHHII64s32sB",
                       b"ACK2",
                       state,
                       last_ack,
                       total_chunks,
                       received_count,
                       bytes_received,
                       bitmask_64,
                       sha_32,
                       err_byte)
    payload_crc = zlib.crc32(body) & 0xffffffff
    payload = body + struct.pack(">I", payload_crc)

    # Q2 Frame Header (16 bytes)
    flags = 0  # Standard frame
    frame_crc = zlib.crc32(payload) & 0xffffffff
    hdr = struct.pack(">2sBBBBHHHI",
                      b"Q2",
                      2,                # Version 2
                      3,                # RGB 8-color mode
                      grid_size,        # Grid dimension (32, 48, 64)
                      flags,
                      0,                # Frame index = 0 (< total)
                      1,                # Total frames = 1
                      len(payload),
                      frame_crc)

    # Convert bytes to 3-bit color cell values
    bpf = (grid_size * grid_size * 3) // 8
    raw_stream = (hdr + payload).ljust(bpf, b"\x00")
    bits = "".join(format(b, "08b") for b in raw_stream)
    vals = [int(bits[i:i + 3], 2) for i in range(0, len(bits) - 2, 3)]
    tot_cells = grid_size * grid_size
    if len(vals) < tot_cells:
        vals.extend([0] * (tot_cells - len(vals)))
    return vals[:tot_cells]


def make_ppm_image(cells, w, h, gs=32):
    """
    Generates a raw PPM (P6) binary image in a pure Python bytearray.
    Used by Tkinter.PhotoImage to display without PIL, OpenCV, or NumPy.
    """
    buf = bytearray(w * h * 3)
    msize_w = max(4, int(w * 0.024))
    msize_h = max(4, int(h * 0.024))

    # 4 Corner Chromatic Fiducials (Red TL, Green TR, Blue BL, Magenta BR)
    fids = (
        (int(w * 0.04), int(h * 0.04), FID_BYTES[0]),
        (int(w * 0.96), int(h * 0.04), FID_BYTES[1]),
        (int(w * 0.04), int(h * 0.96), FID_BYTES[2]),
        (int(w * 0.96), int(h * 0.96), FID_BYTES[3]),
    )
    for cx, cy, col in fids:
        # Outer White Box
        for y in range(max(0, cy - msize_h), min(h, cy + msize_h)):
            row_off = y * w * 3
            for x in range(max(0, cx - msize_w), min(w, cx + msize_w)):
                buf[row_off + x * 3 : row_off + x * 3 + 3] = b"\xff\xff\xff"
        # Middle Black Ring
        rw, rh = max(1, msize_w // 3), max(1, msize_h // 3)
        for y in range(max(0, cy - msize_h + rh), min(h, cy + msize_h - rh)):
            row_off = y * w * 3
            for x in range(max(0, cx - msize_w + rw), min(w, cx + msize_w - rw)):
                buf[row_off + x * 3 : row_off + x * 3 + 3] = b"\x00\x00\x00"
        # Inner Colored Core
        cw, ch_c = max(2, msize_w * 2 // 3), max(2, msize_h * 2 // 3)
        for y in range(max(0, cy - msize_h + ch_c), min(h, cy + msize_h - ch_c)):
            row_off = y * w * 3
            for x in range(max(0, cx - msize_w + cw), min(w, cx + msize_w - cw)):
                buf[row_off + x * 3 : row_off + x * 3 + 3] = col

    # Centered Data Grid (from 8% to 92% of dimension)
    gx0 = int(w * 0.08)
    gy0 = int(h * 0.08)
    gw = max(gs, int(w * 0.84))
    gh = max(gs, int(h * 0.84))

    col_map = [c * gs // gw for c in range(gw)]
    row_bytes_list = []
    for cr in range(gs):
        row_cells = [PAL_BYTES[cells[cr * gs + col_map[c]]] for c in range(gw)]
        row_bytes_list.append(b"".join(row_cells))

    slice_len = gw * 3
    for r in range(gh):
        cr = r * gs // gh
        row_start = (gy0 + r) * w * 3 + gx0 * 3
        buf[row_start : row_start + slice_len] = row_bytes_list[cr]

    return f"P6\n{w} {h}\n255\n".encode("ascii") + bytes(buf)


def render_tty_grid(cells, gs=32):
    """
    Renders the optical grid to a terminal screen using 24-bit ANSI escape codes.
    """
    M = max(2, int(round(gs / 21.0)))
    D = gs + 2 * M
    w = D + 5
    buf = [[ANSI_BLACK_BLOCK] * w for _ in range(w)]

    fids = ((2, 2, 0), (2, 2 + D, 1), (2 + D, 2, 2), (2 + D, 2 + D, 3))
    for cr, cc, fid in fids:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                buf[cr + dr][cc + dc] = ANSI_WHITE_BLOCK
        buf[cr][cc] = ANSI_FIDS[fid]

    r_start = 2 + M
    c_start = 2 + M
    for r in range(gs):
        for c in range(gs):
            buf[r_start + r][c_start + c] = ANSI_RGB_BLOCKS[cells[r * gs + c]]

    return "\033[H" + "\n".join("".join(row) + ANSI_RESET for row in buf) + "\n"


# ==============================================================================
# KEYBOARD PACKET PARSER (Alphanumeric Only)
# ==============================================================================

def _parse_single_packet(line):
    line = line.strip()
    if not line:
        return None

    ptype = line[0].lower()
    content = line[1:]

    try:
        if ptype == "h":
            # Header packet: h<raw_size:08x><comp_size:08x><tot:04x><sha256:64s><namelen:02x><namehex><crc:08x>
            if len(content) < 8 + 8 + 4 + 64 + 2 + 8:
                return None
            body = content[:-8]
            crc_str = content[-8:]
            expected_crc = zlib.crc32(body.encode("ascii")) & 0xffffffff
            if int(crc_str, 16) != expected_crc:
                return None

            raw_size = int(body[0:8], 16)
            comp_size = int(body[8:16], 16)
            tot_chunks = int(body[16:20], 16)
            sha256_hex = body[20:84].lower()
            fn_len = int(body[84:86], 16)
            fn_hex = body[86:86 + fn_len * 2]
            filename = bytes.fromhex(fn_hex).decode("utf-8", errors="replace")
            return ("header", {
                "raw_size": raw_size,
                "comp_size": comp_size,
                "total_chunks": tot_chunks,
                "sha256": sha256_hex,
                "filename": filename
            })

        elif ptype == "d":
            # Data chunk packet: d<idx:04x><crc32:08x><plen:04x or 02x><hex_data>
            if len(content) < 4 + 8 + 2:
                return None
            idx = int(content[0:4], 16)
            crc = int(content[4:12], 16)
            # Try 4-digit hex plen first (supports chunks up to 64 KB)
            if len(content) >= 16:
                plen = int(content[12:16], 16)
                hex_data = content[16:]
                if len(hex_data) == plen * 2:
                    pbytes = bytes.fromhex(hex_data)
                    if (zlib.crc32(pbytes) & 0xffffffff) == crc:
                        return ("data", {"idx": idx, "data": pbytes})

            # Fallback to 2-digit hex plen
            plen = int(content[12:14], 16)
            hex_data = content[14:]
            if len(hex_data) != plen * 2:
                return None
            pbytes = bytes.fromhex(hex_data)
            if (zlib.crc32(pbytes) & 0xffffffff) != crc:
                return None
            return ("data", {
                "idx": idx,
                "data": pbytes
            })

        elif ptype == "m":
            # Chat message: m<msglen:04x><hex_data><crc32:08x>
            if len(content) < 4 + 8:
                return None
            body = content[:-8]
            crc_str = content[-8:]
            expected_crc = zlib.crc32(body.encode("ascii")) & 0xffffffff
            if int(crc_str, 16) != expected_crc:
                return None
            msg_len = int(body[0:4], 16)
            hex_data = body[4:4 + msg_len * 2]
            msg = bytes.fromhex(hex_data).decode("utf-8", errors="replace")
            return ("message", msg)

        elif ptype == "e":
            # End of transmission: e<sha256:64s><crc32:08x>
            if len(content) < 64 + 8:
                return None
            sha256_hex = content[0:64].lower()
            crc_str = content[64:72]
            expected_crc = zlib.crc32(sha256_hex.encode("ascii")) & 0xffffffff
            if int(crc_str, 16) != expected_crc:
                return None
            return ("end", sha256_hex)

        elif ptype == "r":
            return ("reset", None)

    except Exception:
        return None

    return None


def parse_keyboard_packet(line):
    """
    Parses an incoming keyboard packet.
    Uses STRICTLY standard alphanumeric characters [0-9a-fA-F] plus packet type:
    - h: File metadata header
    - d: Data chunk
    - m: Chat message
    - e: End of stream & trigger verification
    - r: Reset

    Recovers automatically if stray prefix characters or concatenated retries exist.
    """
    line = line.strip()
    if not line:
        return None

    pkt = _parse_single_packet(line)
    if pkt is not None:
        return pkt

    # If direct parse failed, scan for rightmost packet markers to strip any partial line prefix
    for marker in ("d", "h", "e", "m", "r"):
        pos = line.rfind(marker)
        if pos > 0:
            pkt = _parse_single_packet(line[pos:])
            if pkt is not None:
                return pkt

    return None


# ==============================================================================
# CLIENT RECEIVER STATE MACHINE
# ==============================================================================

class ClientReceiver:
    def __init__(self, output_dir="./received", grid_size=32):
        self.output_dir = output_dir
        self.grid_size = grid_size
        os.makedirs(self.output_dir, exist_ok=True)

        self.lock = threading.Lock()
        self.state = STATE_WAIT_FOCUS
        self.focus_seq = 0
        self.file_meta = None
        self.chunks = []
        self.received_mask = bytearray(64)
        self.received_count = 0
        self.bytes_received = 0
        self.last_ack = 0
        self.error_code = 0
        self.chat_history = []
        self.saved_filepath = None
        self.actual_sha256 = ""
        self.status_message = "Press ENTER or SPACE when window is focused"

    def confirm_focus(self):
        """Signals keyboard focus to server via optical grid."""
        with self.lock:
            self.state = STATE_FOCUSED
            self.focus_seq = (self.focus_seq + 1) & 0x0F
            self.status_message = f"Focus confirmed (seq={self.focus_seq})! Optical grid ready. Waiting for server..."
            return True

    def reset(self):
        with self.lock:
            self.state = STATE_WAIT_FOCUS
            self.focus_seq = 0
            self.file_meta = None
            self.chunks = []
            self.received_mask = bytearray(64)
            self.received_count = 0
            self.bytes_received = 0
            self.last_ack = 0
            self.error_code = 0
            self.saved_filepath = None
            self.actual_sha256 = ""
            self.status_message = "Reset. Press ENTER or SPACE to signal focus."

    def process_packet(self, pkt):
        if not pkt:
            return

        ptype, payload = pkt
        with self.lock:
            if ptype == "header":
                self.file_meta = payload
                self.chunks = [None] * payload["total_chunks"]
                self.received_mask = bytearray(64)
                self.received_count = 0
                self.bytes_received = 0
                self.last_ack = 0
                self.error_code = 0
                self.state = STATE_RECEIVING
                self.status_message = f"Receiving: {payload['filename']} ({payload['total_chunks']} chunks)"

            elif ptype == "data":
                idx = payload["idx"]
                data = payload["data"]
                if self.file_meta and 0 <= idx < len(self.chunks):
                    if self.chunks[idx] is None:
                        self.chunks[idx] = data
                        self.received_count += 1
                        self.bytes_received += len(data)
                        byte_pos = idx // 8
                        bit_pos = idx % 8
                        if byte_pos < len(self.received_mask):
                            self.received_mask[byte_pos] |= (1 << bit_pos)

                    self.last_ack = idx
                    self.state = STATE_RECEIVING
                    pct = (self.received_count / max(1, len(self.chunks))) * 100.0
                    self.status_message = f"Receiving: {self.received_count}/{len(self.chunks)} chunks ({pct:.1f}%)"

            elif ptype == "message":
                self.chat_history.append(payload)
                if len(self.chat_history) > 6:
                    self.chat_history.pop(0)
                self.status_message = f"[Chat] {payload}"

            elif ptype == "end":
                if self.file_meta and self.received_count == len(self.chunks) and all(c is not None for c in self.chunks):
                    try:
                        comp_data = b"".join(self.chunks)
                        raw_data = zlib.decompress(comp_data)
                        actual_sha = hashlib.sha256(raw_data).hexdigest()
                        self.actual_sha256 = actual_sha

                        if actual_sha == self.file_meta["sha256"]:
                            out_path = os.path.join(self.output_dir, self.file_meta["filename"])
                            with open(out_path, "wb") as f:
                                f.write(raw_data)
                            self.saved_filepath = out_path
                            self.state = STATE_COMPLETE
                            self.status_message = f"SUCCESS! Saved: {self.file_meta['filename']} ({len(raw_data)} bytes)"
                        else:
                            self.state = STATE_ERROR
                            self.error_code = 3  # SHA mismatch
                            self.status_message = "ERROR: SHA-256 mismatch!"
                    except Exception as e:
                        self.state = STATE_ERROR
                        self.error_code = 2  # Decompression error
                        self.status_message = f"ERROR: Decompression failed ({e})"
                else:
                    self.state = STATE_ERROR
                    self.error_code = 1  # Missing chunks
                    self.status_message = "ERROR: End of stream received but chunks are missing"

            elif ptype == "reset":
                self.reset()

    def get_optical_grid(self):
        with self.lock:
            sha_bytes = bytes.fromhex(self.actual_sha256) if self.actual_sha256 else (
                bytes.fromhex(self.file_meta["sha256"]) if self.file_meta else b""
            )
            tot = self.file_meta["total_chunks"] if self.file_meta else 0
            return pack_ack_frame(
                state=self.state,
                last_ack=self.last_ack,
                total_chunks=tot,
                received_count=self.received_count,
                bytes_received=self.bytes_received,
                bitmask_bytes=bytes(self.received_mask),
                sha256_digest=sha_bytes,
                error_code=self.error_code,
                focus_seq=self.focus_seq,
                grid_size=self.grid_size
            )


# ==============================================================================
# RUNNERS: TKINTER GUI & HEADLESS TTY (Standard Library Only)
# ==============================================================================

def stdin_reader_thread(key_queue, stop_event):
    """
    Reads non-blocking characters from terminal standard input.
    Detects Enter or Space immediately even without full lines,
    allowing focus confirmation directly from the terminal window.
    """
    fd = None
    old_attr = None
    try:
        import termios, tty
        if sys.stdin.isatty():
            fd = sys.stdin.fileno()
            old_attr = termios.tcgetattr(fd)
            tty.setcbreak(fd)
    except Exception:
        pass

    line_buf = ""
    try:
        while not stop_event.is_set():
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.02)
                if not r:
                    continue

                if fd is not None:
                    try:
                        chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
                    except Exception:
                        chunk = ""
                else:
                    chunk = sys.stdin.read(1)

                if not chunk:
                    time.sleep(0.01)
                    continue

                for ch in chunk:
                    if ch == "\x03":  # Ctrl+C
                        key_queue.put(("exit", None))
                        return

                    if ch in ("\r", "\n"):
                        trimmed = line_buf.strip()
                        if not trimmed:
                            key_queue.put(("focus", None))
                        else:
                            key_queue.put(("line", trimmed))
                        line_buf = ""
                    elif ch == " ":
                        trimmed = line_buf.strip()
                        if not trimmed:
                            key_queue.put(("focus", None))
                            line_buf = ""
                        else:
                            line_buf += " "
                    elif 32 <= ord(ch) <= 126:
                        line_buf += ch
                        if len(line_buf) > 65536:
                            line_buf = line_buf[-32768:]
            except Exception:
                time.sleep(0.02)
    finally:
        if fd is not None and old_attr is not None:
            try:
                import termios
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
            except Exception:
                pass


def run_tkinter_gui(client):
    """
    Displays the visual grid in a Tkinter GUI window (Python standard library).
    Captures keystrokes from window events and background stdin.
    """
    import tkinter as tk

    init_dim = 600
    root = tk.Tk()
    root.title("otd Duplex Client | Optical Feedback Grid")
    root.configure(bg="#0d1117")
    root.resizable(True, True)
    root.minsize(300, 350)

    # Top Status Banner
    lbl_title = tk.Label(root, text="otd Duplex Client | [WAITING FOR FOCUS]",
                         fg="#f59e0b", bg="#161b22", font=("Monospace", 10, "bold"), pady=6)
    lbl_title.pack(side="top", fill="x")

    canvas = tk.Canvas(root, width=init_dim, height=init_dim, bg="black", highlightthickness=0)
    canvas.pack(side="top", fill="both", expand=True, padx=8, pady=6)

    # Bottom Status Banner
    lbl_status = tk.Label(root, text=client.status_message,
                          fg="#e6edf3", bg="#161b22", font=("Monospace", 9), pady=6)
    lbl_status.pack(side="bottom", fill="x")

    state_obj = {"w": init_dim, "h": init_dim, "photo": None, "line_buf": ""}
    img_id = canvas.create_image(0, 0, anchor="nw")

    key_queue = queue.Queue()
    stop_event = threading.Event()
    t_stdin = threading.Thread(target=stdin_reader_thread, args=(key_queue, stop_event), daemon=True)
    t_stdin.start()

    def render():
        cw = max(50, state_obj["w"])
        ch = max(50, state_obj["h"])
        cells = client.get_optical_grid()
        ppm_data = make_ppm_image(cells, cw, ch, gs=client.grid_size)
        state_obj["photo"] = tk.PhotoImage(data=ppm_data)
        canvas.itemconfig(img_id, image=state_obj["photo"])
        canvas.coords(img_id, 0, 0)

        # Update Header Banner
        st = client.state
        st_txt = STATE_LABELS.get(st, "UNKNOWN")
        lbl_title.config(text=f"otd Duplex Client | [{st_txt}]")
        if st == STATE_WAIT_FOCUS:
            lbl_title.config(fg="#f59e0b")
        elif st == STATE_FOCUSED:
            lbl_title.config(fg="#06b6d4")
        elif st == STATE_RECEIVING:
            lbl_title.config(fg="#3b82f6")
        elif st == STATE_COMPLETE:
            lbl_title.config(fg="#10b981")
        elif st == STATE_ERROR:
            lbl_title.config(fg="#f43f5e")

        # Update Status Banner
        lbl_status.config(text=client.status_message)
        try:
            root.update_idletasks()
        except Exception:
            pass

    render()

    def on_resize(event):
        if event.width > 30 and event.height > 30:
            if event.width != state_obj["w"] or event.height != state_obj["h"]:
                state_obj["w"] = event.width
                state_obj["h"] = event.height
                render()

    canvas.bind("<Configure>", on_resize)

    def on_key(event):
        # 1. Return / Enter
        if event.keysym == "Return" or event.char in ("\r", "\n"):
            line = state_obj["line_buf"].strip()
            state_obj["line_buf"] = ""
            if line:
                if client.state in (STATE_WAIT_FOCUS, STATE_COMPLETE, STATE_ERROR):
                    client.confirm_focus()
                pkt = parse_keyboard_packet(line)
                if pkt:
                    client.process_packet(pkt)
                render()
            else:
                if client.state != STATE_RECEIVING:
                    client.confirm_focus()
                    print(f"\n[Client] Focus confirmed from GUI window (seq={client.focus_seq})! Optical grid ready.")
                    render()
            return

        # 2. Space key
        if event.keysym == "space" or event.char == " ":
            line = state_obj["line_buf"].strip()
            if not line:
                if client.state != STATE_RECEIVING:
                    client.confirm_focus()
                    print(f"\n[Client] Focus confirmed from GUI window (seq={client.focus_seq})! Optical grid ready.")
                    render()
            else:
                state_obj["line_buf"] += " "
            return

        # 3. Regular printable keystrokes
        if event.char and 32 <= ord(event.char) <= 126:
            state_obj["line_buf"] += event.char

    root.bind("<Key>", on_key)

    def check_queue():
        updated = False
        while not key_queue.empty():
            kind, val = key_queue.get_nowait()
            if kind == "exit":
                root.destroy()
                return
            elif kind == "focus":
                if client.state != STATE_RECEIVING:
                    client.confirm_focus()
                    print(f"\n[Client] Focus confirmed from terminal window (seq={client.focus_seq})! Optical grid ready.")
                    updated = True
            elif kind == "line":
                if client.state in (STATE_WAIT_FOCUS, STATE_COMPLETE, STATE_ERROR):
                    client.confirm_focus()
                    updated = True
                pkt = parse_keyboard_packet(val)
                if pkt:
                    client.process_packet(pkt)
                    updated = True

        if updated:
            render()
        root.after(30, check_queue)

    root.after(30, check_queue)

    try:
        root.mainloop()
    finally:
        stop_event.set()


def run_tty(client):
    """
    Runs in headless terminal mode with 24-bit ANSI color grid.
    Pure standard library only (sys, os, termios, tty).
    """
    fd, old_attr = None, None
    try:
        import termios, tty
        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except Exception:
        pass

    sys.stdout.write("\033[?1049h\033[?25l\033[2J")
    sys.stdout.flush()

    line_buf = ""
    try:
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.03)
            if r:
                if fd is not None:
                    try:
                        chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
                    except Exception:
                        chunk = ""
                else:
                    chunk = sys.stdin.read(1)

                for ch in chunk:
                    if ch in ("\x03", "\x1b"):  # Ctrl+C or Esc
                        return
                    elif ch in ("\r", "\n"):
                        trimmed = line_buf.strip()
                        line_buf = ""
                        if trimmed:
                            if client.state in (STATE_WAIT_FOCUS, STATE_COMPLETE, STATE_ERROR):
                                client.confirm_focus()
                            pkt = parse_keyboard_packet(trimmed)
                            if pkt:
                                client.process_packet(pkt)
                        else:
                            if client.state != STATE_RECEIVING:
                                client.confirm_focus()
                    elif ch == " ":
                        trimmed = line_buf.strip()
                        if not trimmed:
                            if client.state != STATE_RECEIVING:
                                client.confirm_focus()
                            line_buf = ""
                        else:
                            line_buf += " "
                    elif 32 <= ord(ch) <= 126:
                        line_buf += ch

            cells = client.get_optical_grid()
            grid_art = render_tty_grid(cells, gs=client.grid_size)
            st_txt = STATE_LABELS.get(client.state, "UNKNOWN")
            status = f"\033[1;33m[otd Duplex Client: {st_txt}]\033[0m {client.status_message}\n"
            sys.stdout.write(grid_art + status)
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\033[?1049l\033[0m")
        sys.stdout.flush()
        if fd is not None and old_attr is not None:
            import termios
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        print("\nClient stopped.")


def run_client(output_dir="./received", grid_size=32, force_tty=False):
    client = ClientReceiver(output_dir=output_dir, grid_size=grid_size)

    print("=" * 65)
    print("   OTD (OPTICAL DATA TRANSFER) DUPLEX CLIENT (STANDALONE)")
    print("   Zero external dependencies (Python 3 standard library only)")
    print("=" * 65)
    print("Instructions:")
    print("  1. In server.py's browser web UI, click 'Share Screen/Window'.")
    print("  2. Set or Auto-Snap the mask around this client grid.")
    print("  3. Select a file to send in the browser.")
    print("  4. CLICK/FOCUS THIS TERMINAL (or GUI window) and press [ENTER] or [SPACE].")
    print("  5. Optical grid will immediately signal focus, and server")
    print("     will automatically type the file directly into this terminal!")
    print("=" * 65)
    print(f"Saving files to: {os.path.abspath(output_dir)}\n")

    if force_tty:
        run_tty(client)
        return

    # Attempt Tkinter GUI first; cleanly fall back to terminal mode if unavailable
    try:
        import tkinter
        run_tkinter_gui(client)
    except Exception as e:
        print(f"Note: GUI display or Tkinter not available ({e}).")
        print("Falling back automatically to pure ANSI terminal mode (--tty)...\n")
        run_tty(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="otd Duplex Client (Pure Python Air-Gap Receiver)")
    parser.add_argument("-o", "--output-dir", default="./received",
                        help="Directory to save received files (default: ./received)")
    parser.add_argument("-g", "--grid", type=int, default=32, choices=[32, 48, 64],
                        help="Optical feedback grid dimension (default: 32)")
    parser.add_argument("--tty", action="store_true",
                        help="Force headless terminal ANSI mode instead of Tkinter GUI")
    args = parser.parse_args()

    run_client(output_dir=args.output_dir, grid_size=args.grid, force_tty=args.tty)
