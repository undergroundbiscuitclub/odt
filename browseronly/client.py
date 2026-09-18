#!/usr/bin/env python3
"""
browseronly/client.py - Enhanced Pure Python Standalone Duplex Client with Active Optical Heartbeat
Zero third-party dependencies (Python 3 standard library only).
Optimized for locked-down RDP/VDI environments, minimal containers, and air-gapped systems.

Enhanced Features for Large Files & Reliable Duplex Links:
1. Active Optical Heartbeat ("Still Waiting" Poll):
   Continuously re-emits structured optical ACK2 frames at 15-20 FPS even when idle or waiting for chunks.
   Prevents server dead-man's switch timeouts ("Optical link interrupted during chunk X/Y") during long transfers.
2. Anti-Stall Visual Pulse:
   Subtle dynamic modulation keeps browser screen-capture (navigator.mediaDevices.getDisplayMedia)
   and desktop compositors streaming active 30-60 FPS video instead of pausing on static windows.
3. Robust Concatenated Packet Parsing:
   Automatically recovers from partial line flushes, terminal buffering, and multi-packet batches.
4. Backward & Forward Compatible:
   Works with browseronly/run.py, server_v2.py, and server.py.
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
# CAUCHY REED-SOLOMON FORWARD ERROR CORRECTION (GF(2^8))
# ==============================================================================

_GF_EXP, _GF_LOG = [0] * 512, [0] * 256
_x = 1
for _i in range(255):
    _GF_EXP[_i] = _x
    _GF_EXP[_i + 255] = _x
    _GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11d
_GF_LOG[0] = 511

def _gf_mul(a, b):
    return 0 if a == 0 or b == 0 else _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]

def _gf_inv(a):
    return _GF_EXP[255 - _GF_LOG[a]]

def _cauchy_coeff(r, c):
    return _gf_inv((255 - r) ^ c)

def generate_fec_parities(chunks, M):
    """Generates M Cauchy Reed-Solomon parity chunks over Galois Field GF(2^8)."""
    K, max_len = len(chunks), max(len(c) for c in chunks)
    parities = []
    for r in range(M):
        p = bytearray(max_len)
        for c in range(K):
            coeff = _cauchy_coeff(r, c)
            d = chunks[c].ljust(max_len, b"\x00")
            for i in range(max_len):
                if d[i]:
                    p[i] ^= _gf_mul(coeff, d[i])
        parities.append(bytes(p))
    return parities

def pack_file_to_frames(filepath, gs=48, fec_pct=15):
    """
    Packs a file into structured Q2 optical frames with Cauchy RS FEC and QSMD metadata.
    100% compatible with browseronly/decoder.js and decoder_v2.py.
    """
    with open(filepath, "rb") as f:
        raw = f.read()

    comp = zlib.compress(raw, 9)
    bn = os.path.basename(filepath).encode("utf-8")
    meta = struct.pack(">4sQQ32sH", b"QSMD", len(raw), len(comp), hashlib.sha256(raw).digest(), len(bn)) + bn
    stream = meta + comp

    bpf = (gs * gs * 3) // 8
    cap = bpf - 16  # 16-byte Q2 header
    chunks = [stream[i:i + cap] for i in range(0, len(stream), cap)]
    data_len = len(chunks)
    all_chunks = list(chunks)

    if fec_pct > 0 and data_len > 0:
        m_count = max(2, int(round(data_len * (fec_pct / 100.0))))
        m_count = min(m_count, max(0, 255 - data_len))
        if m_count > 0:
            all_chunks.extend(generate_fec_parities(chunks, m_count))

    tot = len(all_chunks)
    frames = []
    tot_cells = gs * gs

    for idx, c in enumerate(all_chunks):
        is_parity = idx >= data_len
        crc = zlib.crc32(c) & 0xffffffff
        flags = (1 if idx == 0 else 0) | (2 if is_parity else 0)
        hdr = struct.pack(">2sBBBBHHHI", b"Q2", 2, 3, gs, flags, idx, data_len, len(c), crc)
        raw_frame = (hdr + c).ljust(bpf, b"\x00")

        bits = "".join(format(b, "08b") for b in raw_frame)
        vals = [int(bits[i:i + 3], 2) for i in range(0, len(bits) - 2, 3)]
        if len(vals) < tot_cells:
            vals.extend([0] * (tot_cells - len(vals)))
        frames.append(vals[:tot_cells])

    return frames, tot, len(raw), len(comp), data_len


# ==============================================================================
# PURE PYTHON OPTICAL FRAME ENCODING (With Extended Heartbeat / Poll)
# ==============================================================================

def pack_ack_frame(state, last_ack, total_chunks, received_count, bytes_received,
                   bitmask_bytes, sha256_digest=b"", error_code=0, focus_seq=0,
                   grid_size=32, heartbeat_seq=0):
    """
    Constructs a valid Q2 visual frame containing structured optical ACK telemetry.
    Compatible with decoder_v2's FiducialTracker, server.py, server_v2.py, and browseronly.
    Includes active heartbeat / poll metadata.
    """
    bitmask_64 = bitmask_bytes.ljust(64, b"\x00")[:64]
    sha_32 = sha256_digest.ljust(32, b"\x00")[:32]
    err_byte = (error_code & 0x0F) | ((focus_seq & 0x0F) << 4)

    # Standard ACK2 Body (114 bytes)
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
    payload = body + struct.pack(">I", payload_crc)  # 118 bytes total

    # Optional Extended Heartbeat Tag (appended in frame stream without altering ACK2 payload CRC)
    # Magic b"POLL" + heartbeat_seq (1 byte) + next_expected (2 bytes)
    next_expected = min(total_chunks - 1, last_ack + 1) if total_chunks > 0 else 0
    poll_tag = struct.pack(">4sBH", b"POLL", heartbeat_seq & 0xFF, next_expected)

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

    # Total bytes per frame = (grid_size * grid_size * 3) // 8
    bpf = (grid_size * grid_size * 3) // 8
    full_payload = hdr + payload + poll_tag
    raw_stream = full_payload.ljust(bpf, b"\x00")

    # Convert bytes to 3-bit color cell values
    bits = "".join(format(b, "08b") for b in raw_stream)
    vals = [int(bits[i:i + 3], 2) for i in range(0, len(bits) - 2, 3)]
    tot_cells = grid_size * grid_size
    if len(vals) < tot_cells:
        vals.extend([0] * (tot_cells - len(vals)))
    return vals[:tot_cells]


def make_ppm_image(cells, w, h, gs=32, pulse_state=0):
    """
    Generates a raw PPM (P6) binary image in a pure Python bytearray.
    Includes dynamic anti-stall visual pulse to keep screen capturers (WebRTC / getDisplayMedia)
    streaming at full frame rates.
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

    # Anti-Stall Visual Pulse Indicator (Top Border between Fiducials)
    # Subtle 12x4 pixel indicator alternating green/cyan/yellow to force screen share video updates
    pulse_cols = [b"\x06\xb6\xd4", b"\x10\xb9\x81", b"\x3b\x82\xf6", b"\xfb\xbf\x24"]
    p_col = pulse_cols[pulse_state % len(pulse_cols)]
    px_center = w // 2
    py_pos = max(2, int(h * 0.03))
    pw, ph = max(6, int(w * 0.025)), max(2, int(h * 0.008))
    for y in range(max(0, py_pos - ph), min(h, py_pos + ph)):
        row_off = y * w * 3
        for x in range(max(0, px_center - pw), min(w, px_center + pw)):
            buf[row_off + x * 3 : row_off + x * 3 + 3] = p_col

    return f"P6\n{w} {h}\n255\n".encode("ascii") + bytes(buf)


def render_tty_grid(cells, gs=32, pulse_state=0):
    """
    Renders the optical grid to a terminal screen using 24-bit ANSI escape codes.
    Includes active visual pulse character.
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

    # Subtle pulse marker at center top
    pulse_glyphs = ["🟢", "🔵", "🟡", "⚪"]
    glyph = pulse_glyphs[pulse_state % len(pulse_glyphs)]

    rendered_grid = "\033[H" + "\n".join("".join(row) + ANSI_RESET for row in buf) + "\n"
    return rendered_grid, glyph


# ==============================================================================
# ROBUST KEYBOARD PACKET PARSER
# ==============================================================================

def _parse_single_packet(content_str):
    line = content_str.strip()
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
                hex_data = content[16:16 + plen * 2]
                if len(hex_data) == plen * 2:
                    pbytes = bytes.fromhex(hex_data)
                    if (zlib.crc32(pbytes) & 0xffffffff) == crc:
                        return ("data", {"idx": idx, "data": pbytes})

            # Fallback to 2-digit hex plen
            plen = int(content[12:14], 16)
            hex_data = content[14:14 + plen * 2]
            if len(hex_data) == plen * 2:
                pbytes = bytes.fromhex(hex_data)
                if (zlib.crc32(pbytes) & 0xffffffff) == crc:
                    return ("data", {"idx": idx, "data": pbytes})

            return None

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
            if content in ("", "eset"):
                return ("reset", None)
            return None

    except Exception:
        return None

    return None


def parse_keyboard_packets(raw_text):
    """
    Extracts ALL valid packets from an incoming buffer.
    Handles concatenated packets, newline separation, and partial retry prefix garbage.
    """
    packets = []
    lines = raw_text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue

        pkt = _parse_single_packet(line)
        if pkt is not None:
            packets.append(pkt)
            continue

        # If direct line parse failed, scan for packet markers with CRC verification (d, h, e, m)
        pos = 0
        while pos < len(line):
            best_idx = -1
            for marker in ("d", "h", "e", "m"):
                idx = line.find(marker, pos)
                if idx != -1 and (best_idx == -1 or idx < best_idx):
                    best_idx = idx

            if best_idx == -1:
                break

            sub = line[best_idx:]
            p = _parse_single_packet(sub)
            if p is not None:
                packets.append(p)
                break
            pos = best_idx + 1

    return packets


# ==============================================================================
# ENHANCED CLIENT RECEIVER STATE MACHINE WITH OPTICAL HEARTBEAT
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

        # Heartbeat & Optical Keep-Alive metrics
        self.heartbeat_counter = 0
        self.last_heartbeat_time = time.time()
        self.last_packet_time = time.time()
        self.pulse_state = 0

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

    def step_heartbeat(self):
        """Advances heartbeat sequence and anti-stall pulse."""
        with self.lock:
            self.heartbeat_counter = (self.heartbeat_counter + 1) & 0xFFFF
            self.pulse_state = (self.pulse_state + 1) & 0xFF
            self.last_heartbeat_time = time.time()

            # If actively receiving but no chunk arrived in last 1.5s, update status
            if self.state == STATE_RECEIVING and self.file_meta:
                dt = time.time() - self.last_packet_time
                if dt > 1.2:
                    next_chunk = self.last_ack + 1
                    tot = len(self.chunks)
                    self.status_message = (
                        f"Optical Poll: Waiting for chunk {next_chunk + 1}/{tot} "
                        f"(Link Live, Heartbeat #{self.heartbeat_counter})"
                    )

    def process_packet(self, pkt):
        if not pkt:
            return

        ptype, payload = pkt
        with self.lock:
            self.last_packet_time = time.time()
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
                    self.status_message = (
                        f"Receiving: {self.received_count}/{len(self.chunks)} chunks ({pct:.1f}%) "
                        f"| Last ACK: #{idx + 1}"
                    )

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
                grid_size=self.grid_size,
                heartbeat_seq=self.heartbeat_counter
            )


# ==============================================================================
# RUNNERS: ACTIVE TKINTER GUI & HEADLESS TTY (With Optical Poll Engine)
# ==============================================================================

def stdin_reader_thread(key_queue, stop_event):
    """Reads non-blocking characters from terminal standard input."""
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
    """Displays the visual grid in an active Tkinter GUI window with optical heartbeat."""
    import tkinter as tk

    init_dim = 600
    root = tk.Tk()
    root.title("OTD Duplex Client (Active Optical Heartbeat)")
    root.configure(bg="#0d1117")
    root.resizable(True, True)
    root.minsize(300, 350)

    # Top Status Banner
    lbl_title = tk.Label(root, text="OTD Duplex Client | [WAITING FOR FOCUS]",
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
        ppm_data = make_ppm_image(cells, cw, ch, gs=client.grid_size, pulse_state=client.pulse_state)
        state_obj["photo"] = tk.PhotoImage(data=ppm_data)
        canvas.itemconfig(img_id, image=state_obj["photo"])
        canvas.coords(img_id, 0, 0)

        # Update Header Banner
        st = client.state
        st_txt = STATE_LABELS.get(st, "UNKNOWN")
        pulse_icon = "🟢" if (client.pulse_state % 2 == 0) else "🔵"
        lbl_title.config(text=f"OTD Duplex Client {pulse_icon} | [{st_txt}]")
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
        if event.keysym == "Return" or event.char in ("\r", "\n"):
            line = state_obj["line_buf"].strip()
            state_obj["line_buf"] = ""
            if line:
                if client.state in (STATE_WAIT_FOCUS, STATE_COMPLETE, STATE_ERROR):
                    client.confirm_focus()
                pkts = parse_keyboard_packets(line)
                for pkt in pkts:
                    client.process_packet(pkt)
                render()
            else:
                if client.state != STATE_RECEIVING:
                    client.confirm_focus()
                    print(f"\n[Client] Focus confirmed from GUI window (seq={client.focus_seq})! Optical grid ready.")
                    render()
            return

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

        if event.char and 32 <= ord(event.char) <= 126:
            state_obj["line_buf"] += event.char

    root.bind("<Key>", on_key)

    # Active Heartbeat & Queue Polling (20 FPS / 50ms interval)
    def active_heartbeat_loop():
        # 1. Process all pending keystroke lines from terminal stdin
        while not key_queue.empty():
            kind, val = key_queue.get_nowait()
            if kind == "exit":
                root.destroy()
                return
            elif kind == "focus":
                if client.state != STATE_RECEIVING:
                    client.confirm_focus()
                    print(f"\n[Client] Focus confirmed from terminal window (seq={client.focus_seq})! Optical grid ready.")
            elif kind == "line":
                if client.state in (STATE_WAIT_FOCUS, STATE_COMPLETE, STATE_ERROR):
                    client.confirm_focus()
                pkts = parse_keyboard_packets(val)
                for pkt in pkts:
                    client.process_packet(pkt)

        # 2. Step heartbeat counter and re-render frame
        client.step_heartbeat()
        render()

        # 3. Schedule next tick (50ms = 20 FPS steady optical broadcast)
        root.after(50, active_heartbeat_loop)

    root.after(50, active_heartbeat_loop)

    try:
        root.mainloop()
    finally:
        stop_event.set()


def run_tty(client):
    """Runs in headless terminal mode with 24-bit ANSI color grid and active optical heartbeat."""
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
    last_draw = 0
    try:
        while True:
            # Check stdin non-blocking
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
                            pkts = parse_keyboard_packets(trimmed)
                            for pkt in pkts:
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

            # Steady 10 FPS TTY heartbeat render
            now = time.time()
            if now - last_draw >= 0.10:
                last_draw = now
                client.step_heartbeat()
                cells = client.get_optical_grid()
                grid_art, glyph = render_tty_grid(cells, gs=client.grid_size, pulse_state=client.pulse_state)
                st_txt = STATE_LABELS.get(client.state, "UNKNOWN")
                status = f"\033[1;33m[OTD Duplex Client {glyph}: {st_txt}]\033[0m {client.status_message}\n"
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


def run_gui_sender(frames, tot, delay_ms, fname, gs=48, data_len=None):
    """Displays the optical transmission frames in an interactive Tkinter GUI window."""
    if data_len is None: data_len = tot
    import tkinter as tk
    init_dim = 650

    root = tk.Tk()
    root.title(f"OTD Optical Transmitter | {fname}")
    root.configure(bg="#000000")
    root.resizable(True, True)
    root.minsize(200, 200)

    # Top Status Bar
    lbl = tk.Label(root, text="BROADCASTING - Press Space to Pause | 'f' Fullscreen",
                   fg="#38bdf8", bg="#0d1117", font=("Monospace", 9, "bold"), pady=6)
    lbl.pack(side="top", fill="x")

    canvas = tk.Canvas(root, width=init_dim, height=init_dim, bg="black", highlightthickness=0)
    canvas.pack(side="top", fill="both", expand=True)

    state = {
        "idx": 0,
        "run": True,
        "loop": 1,
        "photo": None,
        "w": init_dim,
        "h": init_dim,
        "fullscreen": False
    }
    img_id = canvas.create_image(0, 0, anchor="nw")

    def render():
        cw = max(50, state["w"])
        ch = max(50, state["h"])
        ppm_data = make_ppm_image(frames[state["idx"]], cw, ch, gs=gs, pulse_state=state["idx"])
        state["photo"] = tk.PhotoImage(data=ppm_data)
        canvas.itemconfig(img_id, image=state["photo"])
        canvas.coords(img_id, 0, 0)

        is_parity = state["idx"] >= data_len
        tag = f"Parity {state['idx'] - data_len + 1}/{tot - data_len}" if is_parity else f"Data {state['idx'] + 1}/{data_len}"
        pct = min(100.0, ((state["idx"] + 1) / data_len) * 100.0) if data_len > 0 else 100.0
        fps_est = 1000.0 / max(1, delay_ms)
        st_text = "BROADCASTING" if state["run"] else "PAUSED"
        color = "#10b981" if state["run"] else "#f59e0b"
        lbl.config(
            text=f"[{st_text}] {fname} | {tag} ({pct:.1f}%) | Loop #{state['loop']} | {fps_est:.0f} FPS | Space: Toggle | 'f': Fullscreen",
            fg=color
        )

    def on_resize(event):
        if event.width > 30 and event.height > 30:
            if event.width != state["w"] or event.height != state["h"]:
                state["w"] = event.width
                state["h"] = event.height
                render()

    canvas.bind("<Configure>", on_resize)

    def toggle(event=None):
        state["run"] = not state["run"]
        render()

    def step_next(event=None):
        state["idx"] = (state["idx"] + 1) % tot
        render()

    def step_prev(event=None):
        state["idx"] = (state["idx"] - 1 + tot) % tot
        render()

    def toggle_fullscreen(event=None):
        state["fullscreen"] = not state["fullscreen"]
        root.attributes("-fullscreen", state["fullscreen"])

    root.bind("<space>", toggle)
    root.bind("<Return>", toggle)
    root.bind("<Button-1>", toggle)
    root.bind("n", step_next)
    root.bind("N", step_next)
    root.bind("<Right>", step_next)
    root.bind("p", step_prev)
    root.bind("P", step_prev)
    root.bind("<Left>", step_prev)
    root.bind("f", toggle_fullscreen)
    root.bind("F", toggle_fullscreen)
    root.bind("q", lambda e: root.destroy())
    root.bind("Q", lambda e: root.destroy())
    root.bind("<Escape>", lambda e: root.destroy())

    def tick():
        if state["run"]:
            state["idx"] = (state["idx"] + 1) % tot
            if state["idx"] == 0:
                state["loop"] += 1
            render()
        root.after(delay_ms, tick)

    render()
    root.after(delay_ms, tick)
    root.mainloop()


def run_tty_sender(frames, tot, delay_ms, fname, gs=48, data_len=None):
    """Runs the optical transmitter in headless terminal mode with 24-bit ANSI color."""
    if data_len is None: data_len = tot
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
    state = {"idx": 0, "run": True, "loop": 1}
    last_t = 0

    def get_ch():
        try:
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
        except Exception:
            pass
        return None

    try:
        while True:
            k = get_ch()
            if k in ("q", "Q", "\x03", "\x1b"):
                break
            elif k in (" ", "\n", "\r"):
                state["run"] = not state["run"]
            elif k in ("n", "N", ">"):
                state["idx"] = (state["idx"] + 1) % tot
            elif k in ("p", "P", "<"):
                state["idx"] = (state["idx"] - 1 + tot) % tot

            now = time.time()
            if (now - last_t) * 1000.0 >= delay_ms or last_t == 0 or not state["run"]:
                pct = min(100.0, ((state["idx"] + 1) / data_len) * 100.0)
                is_parity = state["idx"] >= data_len
                tag = f"Parity {state['idx'] - data_len + 1:2d}/{tot - data_len}" if is_parity else f"Frame {state['idx'] + 1:3d}/{data_len:3d}"
                st = "BROADCASTING" if state["run"] else "PAUSED"
                fps = 1000.0 / max(1, delay_ms)
                status = f"\033[1;33m[{st}] {fname} | {tag} ({pct:5.1f}%) | Loop #{state['loop']} | {fps:.0f} FPS | Space: Toggle | q: Quit\033[0m\n"
                grid_art, _ = render_tty_grid(frames[state["idx"]], gs=gs, pulse_state=state["idx"])
                sys.stdout.write(grid_art + status)
                sys.stdout.flush()
                last_t = now

                if state["run"]:
                    state["idx"] = (state["idx"] + 1) % tot
                    if state["idx"] == 0:
                        state["loop"] += 1
            time.sleep(0.005)
    finally:
        sys.stdout.write("\033[?25h\033[?1049l\033[0m")
        sys.stdout.flush()
        if fd is not None and old_attr is not None:
            import termios
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        print("\nOptical Transmitter stopped.")


def run_send(filepath, grid_size=48, delay_ms=60, fec_pct=15, force_tty=False):
    """Encodes a file and optically broadcasts it onto screen (GUI or TTY)."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    print("=" * 68)
    print("   OTD OPTICAL TRANSMITTER / SENDER (STANDALONE)")
    print("   Zero external dependencies (Python 3 standard library only)")
    print("=" * 68)
    frames, tot, osize, csize, data_len = pack_file_to_frames(filepath, gs=grid_size, fec_pct=fec_pct)
    parity_count = tot - data_len
    fec_str = f" (+{parity_count} Cauchy FEC parity)" if parity_count > 0 else ""
    fps = 1000.0 / max(1, delay_ms)
    fname = os.path.basename(filepath)

    print(f"File:       {fname}")
    print(f"Size:       {osize:,} bytes (compressed to {csize:,} bytes)")
    print(f"Frames:     {tot} total{fec_str}")
    print(f"Grid:       {grid_size}×{grid_size} (RGB 8-color mode, 3 bits/cell)")
    print(f"Speed:      {delay_ms} ms/frame (~{fps:.1f} FPS)")
    print("=" * 68)
    print("Point the camera or screen share from browseronly/index.html at this window!\n")

    if force_tty:
        run_tty_sender(frames, tot, delay_ms, fname, gs=grid_size, data_len=data_len)
        return

    try:
        import tkinter
        run_gui_sender(frames, tot, delay_ms, fname, gs=grid_size, data_len=data_len)
    except Exception as e:
        print(f"Note: Tkinter GUI not available ({e}). Falling back to ANSI terminal mode (--tty)...\n")
        run_tty_sender(frames, tot, delay_ms, fname, gs=grid_size, data_len=data_len)


def run_client(output_dir="./received", grid_size=32, force_tty=False):
    client = ClientReceiver(output_dir=output_dir, grid_size=grid_size)

    print("=" * 68)
    print("   OTD DUPLEX CLIENT (WITH ACTIVE OPTICAL HEARTBEAT / POLL)")
    print("   Zero external dependencies (Python 3 standard library only)")
    print("=" * 68)
    print("Features:")
    print("  • Active Optical Heartbeat keeps dead-man's watchdogs alive on big files.")
    print("  • Anti-Stall Visual Pulse forces continuous 30-60 FPS screen capture.")
    print("  • Robust multi-packet parser prevents dropped chunks on fast typing.")
    print("=" * 68)
    print("Instructions:")
    print("  1. In browseronly/index.html, click 'Share Screen' or 'Use Camera'.")
    print("  2. Focus this window and press [ENTER] or [SPACE].")
    print("  3. The optical grid will confirm focus and begin receiving!")
    print("=" * 68)
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
    parser = argparse.ArgumentParser(
        description="OTD Standalone Duplex Client: Optical Receiver & Transmitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Default Receiver Mode (read keystrokes & show optical ACK grid):
  python3 browseronly/client.py

  # Transmit / Send Mode (optically broadcast a file to camera or screen share):
  python3 browseronly/client.py --send /path/to/file.txt --delay 60 --grid 48

  # Headless ANSI terminal transmitter:
  python3 browseronly/client.py --send /path/to/file.txt --tty --fps 20
"""
    )
    parser.add_argument("-s", "--send", type=str, default=None, metavar="FILE",
                        help="Transmit a file optically (Optical Sender / Broadcast Mode)")
    parser.add_argument("-d", "--delay", type=int, default=None,
                        help="Frame delay in milliseconds for optical transmission (default: 60ms)")
    parser.add_argument("--fps", type=int, default=None,
                        help="Playback framerate in frames per second (e.g. 20, 30; overrides --delay)")
    parser.add_argument("-g", "--grid", type=int, default=None, choices=[32, 48, 64, 80, 96],
                        help="Optical grid dimension (default: 48 for --send, 32 for receive)")
    parser.add_argument("-f", "--fec", type=int, default=15,
                        help="Cauchy Reed-Solomon FEC parity percentage (default: 15)")
    parser.add_argument("-o", "--output-dir", default="./received",
                        help="Directory to save received files when in receive mode (default: ./received)")
    parser.add_argument("--tty", action="store_true",
                        help="Force headless terminal ANSI mode instead of Tkinter GUI")
    parser.add_argument("--b32", "--decode-base32", type=str, default=None, metavar="OUTFILE",
                        help="Decode continuous Base32 keystrokes from stdin directly into OUTFILE")
    args = parser.parse_args()

    if args.b32:
        import base64
        outfile = args.b32
        print(f"[Client] Reading Base32 stream from stdin -> writing to '{outfile}'...")
        raw_text = sys.stdin.read()
        cleaned = "".join(raw_text.split())
        if not cleaned:
            print("[Client Error] No Base32 input received.")
            sys.exit(1)
        padding_needed = (8 - len(cleaned) % 8) % 8
        cleaned += "=" * padding_needed
        try:
            data = base64.b32decode(cleaned.encode("ascii"), casefold=True)
            with open(outfile, "wb") as f:
                f.write(data)
            print(f"[Client] Successfully wrote {len(data)} bytes to '{outfile}'!")
        except Exception as e:
            print(f"[Client Error] Failed to decode Base32: {e}")
            sys.exit(1)
    elif args.send:
        # Determine delay
        if args.fps and args.fps > 0:
            delay = max(1, 1000 // args.fps)
        elif args.delay and args.delay > 0:
            delay = args.delay
        else:
            delay = 60  # Default ~16.6 FPS

        gs = args.grid if args.grid else 48
        run_send(args.send, grid_size=gs, delay_ms=delay, fec_pct=args.fec, force_tty=args.tty)
    else:
        gs = args.grid if args.grid else 32
        run_client(output_dir=args.output_dir, grid_size=gs, force_tty=args.tty)
