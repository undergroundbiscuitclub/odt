#!/usr/bin/env python3
"""
encoder_v2.py - High-Speed Visual Optical Transmitter (v2)
Supports both GUI Window mode (Tkinter) and Headless Shell mode (--tty).
Optimized for high-throughput file transfer (up to 20MB+) over RDP sessions, SSH, and screen recordings.
Uses 8-color RGB multiplexing (3 bits/cell), 4-corner perspective fiducials, and zlib level 9 compression.
"""

import argparse
import hashlib
import os
import select
import struct
import sys
import time
import zlib
import numpy as np

# 8 Saturated RGB Corner Colors (3 bits per cell)
# Format: RGB
PALETTE_RGB = [
    (0, 0, 0),       # 000: Black
    (0, 0, 255),     # 001: Blue
    (0, 255, 0),     # 010: Green
    (0, 255, 255),   # 011: Cyan
    (255, 0, 0),     # 100: Red
    (255, 0, 255),   # 101: Magenta
    (255, 255, 0),   # 110: Yellow
    (255, 255, 255)  # 111: White
]

PALETTE_BW = [
    (0, 0, 0),       # 0: Black
    (255, 255, 255)  # 1: White
]

PALETTE_RGB_NP = np.array(PALETTE_RGB, dtype=np.uint8)
PALETTE_BW_NP = np.array(PALETTE_BW, dtype=np.uint8)

# Precomputed ANSI escape blocks for fast terminal output (2 spaces per cell)
ANSI_RGB_BLOCKS = [f"\033[48;2;{r};{g};{b}m  " for r, g, b in PALETTE_RGB]
ANSI_BW_BLOCKS = [f"\033[48;2;{r};{g};{b}m  " for r, g, b in PALETTE_BW]

ANSI_WHITE_BLOCK = "\033[48;2;255;255;255m  "
ANSI_BLACK_BLOCK = "\033[48;2;0;0;0m  "
ANSI_RED_BLOCK = "\033[48;2;255;0;0m  "
ANSI_GREEN_BLOCK = "\033[48;2;0;255;0m  "
ANSI_BLUE_BLOCK = "\033[48;2;0;0;255m  "
ANSI_MAGENTA_BLOCK = "\033[48;2;255;0;255m  "
ANSI_RESET = "\033[0m"

HEADER_SIZE = 16  # 16 bytes frame header

def prepare_stream(filepath):
    with open(filepath, "rb") as f:
        raw_data = f.read()

    orig_size = len(raw_data)
    orig_sha = hashlib.sha256(raw_data).digest()
    compressed = zlib.compress(raw_data, level=9)
    comp_size = len(compressed)

    basename = os.path.basename(filepath).encode("utf-8")
    # Metadata block in Chunk 0:
    # Magic(4B: b"QSMD"), OrigSize(8B), CompSize(8B), SHA256(32B), FnameLen(2B), Filename(N B)
    meta_header = struct.pack(">4sQQ32sH", b"QSMD", orig_size, comp_size, orig_sha, len(basename)) + basename
    full_stream = meta_header + compressed

    return full_stream, orig_size, comp_size, orig_sha

# =====================================================================
# FORWARD ERROR CORRECTION (CAUCHY REED-SOLOMON GF(2^8) ERASURE CODE)
# =====================================================================
_GF_EXP = [0] * 512
_GF_LOG = [0] * 256
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
    K = len(chunks)
    max_len = max(len(c) for c in chunks)
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

def pack_frames(full_stream, grid_size, mode, fec_pct=15):
    bits_per_cell = 3 if mode == "rgb" else 1
    total_cells = grid_size * grid_size
    total_bits_per_frame = total_cells * bits_per_cell
    total_bytes_per_frame = total_bits_per_frame // 8
    payload_capacity = total_bytes_per_frame - HEADER_SIZE

    chunks = [full_stream[i:i + payload_capacity] for i in range(0, len(full_stream), payload_capacity)]
    data_chunks_len = len(chunks)
    all_chunks = list(chunks)

    if fec_pct > 0 and data_chunks_len > 0:
        m_count = max(2, int(round(data_chunks_len * (fec_pct / 100.0))))
        m_count = min(m_count, max(0, 255 - data_chunks_len))
        if m_count > 0:
            parities = generate_fec_parities(chunks, m_count)
            all_chunks.extend(parities)

    mode_byte = 3 if mode == "rgb" else 1
    frames_data = []

    for idx, chunk in enumerate(all_chunks):
        is_parity = idx >= data_chunks_len
        crc = zlib.crc32(chunk) & 0xffffffff
        flags = (1 if idx == 0 else 0) | (2 if is_parity else 0)
        header = struct.pack(">2sBBBBHHHI", b"Q2", 2, mode_byte, grid_size, flags, idx, data_chunks_len, len(chunk), crc)
        raw_frame = (header + chunk).ljust(total_bytes_per_frame, b"\x00")

        # Convert to cell values
        bits = "".join(format(b, "08b") for b in raw_frame)
        if mode == "rgb":
            cell_vals = [int(bits[i:i + 3], 2) for i in range(0, len(bits) - 2, 3)]
        else:
            cell_vals = [int(b) for b in bits]

        if len(cell_vals) < total_cells:
            cell_vals.extend([0] * (total_cells - len(cell_vals)))

        grid = np.array(cell_vals[:total_cells], dtype=np.uint8).reshape((grid_size, grid_size))
        frames_data.append((idx, data_chunks_len, grid))

    return frames_data, payload_capacity

# =====================================================================
# GUI TRANSFORMATION & RENDERING (Tkinter)
# =====================================================================

def render_frame_image(idx, total_chunks, grid_data, canvas_w, canvas_h=None, grid_size=64, mode="rgb", loop_num=1):
    """
    Renders the visual frame as an RGB NumPy array for GUI display.
    Supports dynamic scaling, arbitrary aspect ratio stretching, and window resizing.
    """
    # Backwards compatibility: if canvas_h looks like cell_px (small int <= 32 or None), treat as square
    if canvas_h is None or (isinstance(canvas_h, int) and canvas_h <= 32):
        w = int(canvas_w)
        h = int(canvas_w)
    else:
        w = int(canvas_w)
        h = int(canvas_h)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    palette = PALETTE_RGB_NP if mode == "rgb" else PALETTE_BW_NP

    # 1. Corner Fiducials
    msize_w = max(4, int(w * 0.024))
    msize_h = max(4, int(h * 0.024))
    fiducials = [
        ("TL", int(w * 0.04), int(h * 0.04), (255, 0, 0)),     # Red
        ("TR", int(w * 0.96), int(h * 0.04), (0, 255, 0)),     # Green
        ("BL", int(w * 0.04), int(h * 0.96), (0, 0, 255)),     # Blue
        ("BR", int(w * 0.96), int(h * 0.96), (255, 0, 255))    # Magenta
    ]

    for _, cx, cy, col in fiducials:
        # Outer White Box
        img[max(0, cy - msize_h):cy + msize_h, max(0, cx - msize_w):cx + msize_w] = [255, 255, 255]
        # Middle Black Ring
        ring_w = max(1, msize_w // 3)
        ring_h = max(1, msize_h // 3)
        img[max(0, cy - msize_h + ring_h):cy + msize_h - ring_h, max(0, cx - msize_w + ring_w):cx + msize_w - ring_w] = [0, 0, 0]
        # Inner Colored Core
        core_w = max(2, msize_w * 2 // 3)
        core_h = max(2, msize_h * 2 // 3)
        img[max(0, cy - msize_h + core_h):cy + msize_h - core_h, max(0, cx - msize_w + core_w):cx + msize_w - core_w] = col

    # 2. Data Grid (from 0.08 to 0.92 of width & height)
    grid_x0 = int(w * 0.08)
    grid_y0 = int(h * 0.08)
    grid_w = max(grid_size, int(w * 0.84))
    grid_h = max(grid_size, int(h * 0.84))

    grid_rgb = palette[grid_data]

    # Vectorized 2D nearest-neighbor stretching
    r_idx = (np.arange(grid_h) * grid_size // grid_h).astype(np.int32)
    c_idx = (np.arange(grid_w) * grid_size // grid_w).astype(np.int32)
    grid_scaled = grid_rgb[r_idx[:, None], c_idx]
    img[grid_y0:grid_y0 + grid_h, grid_x0:grid_x0 + grid_w] = grid_scaled

    # 3. Bottom Progress Bar
    bar_y1 = int(h * 0.945)
    bar_y2 = int(h * 0.960)
    bar_x1 = grid_x0
    bar_x2 = grid_x0 + grid_w
    bar_width = max(1, bar_x2 - bar_x1)

    img[bar_y1:bar_y2, bar_x1:bar_x2] = [40, 40, 40]
    fill_w = min(bar_width, int(((idx + 1) / total_chunks) * bar_width))
    if fill_w > 0:
        bar_col = [200, 50, 255] if idx >= total_chunks else [0, 255, 0]
        img[bar_y1:bar_y2, bar_x1:bar_x1 + fill_w] = bar_col

    # 4. First frame loop flash indicator
    if idx == 0:
        border_w = max(1, min(msize_w, msize_h) // 4)
        img[grid_y0 - border_w:grid_y0, grid_x0 - border_w:grid_x0 + grid_w + border_w] = [255, 255, 0]
        img[grid_y0 + grid_h:grid_y0 + grid_h + border_w, grid_x0 - border_w:grid_x0 + grid_w + border_w] = [255, 255, 0]
        img[grid_y0 - border_w:grid_y0 + grid_h + border_w, grid_x0 - border_w:grid_x0] = [255, 255, 0]
        img[grid_y0 - border_w:grid_y0 + grid_h + border_w, grid_x0 + grid_w:grid_x0 + grid_w + border_w] = [255, 255, 0]

    return img

def run_gui_transmitter(filepath, grid_size=64, cell_px=None, delay_ms=70, mode="rgb", fec_pct=15):
    """Runs the visual transmitter in a Tkinter GUI window."""
    try:
        import tkinter as tk
    except ImportError:
        print("Note: Tkinter is not available in this environment. Falling back to --tty mode...\n")
        run_shell_transmitter(filepath, grid_size=grid_size, delay_ms=delay_ms, mode=mode, fec_pct=fec_pct)
        return

    full_stream, orig_size, comp_size, orig_sha = prepare_stream(filepath)
    frames_data, payload_cap = pack_frames(full_stream, grid_size, mode, fec_pct=fec_pct)
    total_frames = len(frames_data)
    data_frames = frames_data[0][1] if frames_data else 0
    parity_frames = total_frames - data_frames

    if cell_px is None:
        cell_px = max(4, min(12, 576 // grid_size))

    grid_dim = grid_size * cell_px
    canvas_dim = int(grid_dim / 0.84)

    fps = 1000.0 / delay_ms
    kb_per_sec = (payload_cap * fps) / 1024.0
    est_sec = (comp_size / (payload_cap * fps)) if (payload_cap * fps) > 0 else 0

    print("=" * 65)
    print("   OTD (OPTICAL DATA TRANSFER) v2 TRANSMITTER (GUI)")
    print("=" * 65)
    print(f"File:           {filepath}")
    print(f"Original Size:  {orig_size:,} bytes ({orig_size / 1024 / 1024:.2f} MB)")
    print(f"Compressed:     {comp_size:,} bytes ({comp_size / 1024 / 1024:.2f} MB, {comp_size/orig_size*100:.1f}%)")
    print(f"SHA-256:        {orig_sha.hex()[:16]}...{orig_sha.hex()[-8:]}")
    print(f"Grid Dimension: {grid_size}x{grid_size} ({'8 Colors RGB' if mode == 'rgb' else 'B/W'})")
    print(f"Cell Size:      {cell_px} px (Canvas: {canvas_dim}x{canvas_dim} px)")
    print(f"Capacity:       {payload_cap:,} bytes/frame")
    print(f"Data Frames:    {data_frames:,} frames")
    if parity_frames > 0:
        print(f"FEC Parity:     +{parity_frames} Cauchy parity frames ({fec_pct}% redundancy)")
    print(f"Total Frames:   {total_frames:,} frames / cycle")
    print(f"Broadcast Rate: {fps:.1f} FPS ({delay_ms} ms delay)")
    print(f"Throughput:     ~{kb_per_sec:.1f} KB/s")
    print(f"1-Pass Time:    ~{est_sec:.1f} seconds")
    print("=" * 65)
    print("Controls: Click 'START' or press [Space]/[Enter] to begin transmission\n")

    try:
        root = tk.Tk()
    except Exception as e:
        print(f"Note: Cannot open display ({e}). Falling back to --tty mode...\n")
        run_shell_transmitter(filepath, grid_size=grid_size, delay_ms=delay_ms, mode=mode)
        return

    root.title(f"otd Transmitter v2 [{os.path.basename(filepath)}]")
    root.configure(bg="black")
    root.resizable(True, True)
    root.minsize(200, 200)

    status_label = tk.Label(root, text="", fg="yellow", bg="black", font=("Monospace", 10, "bold"))
    status_label.pack(side="top", fill="x", pady=2)

    # Bottom Control Bar
    ctrl_frame = tk.Frame(root, bg="black")
    ctrl_frame.pack(side="bottom", fill="x", padx=4, pady=2)

    # Canvas fills available area and scales dynamically
    canvas = tk.Canvas(root, width=canvas_dim, height=canvas_dim, bg="black", highlightthickness=0)
    canvas.pack(side="top", fill="both", expand=True, padx=4, pady=2)

    canvas_dims = {"w": canvas_dim, "h": canvas_dim}

    state = {
        "idx": 0,
        "count": 0,
        "started": False,
        "paused": False,
        "delay": delay_ms,
        "photo": None,
        "after_id": None
    }

    def render_current_frame():
        idx, total, grid_data = frames_data[state["idx"]]
        loop_num = (state["count"] // total) + 1 if state["started"] else 1
        cw = max(50, canvas_dims["w"])
        ch = max(50, canvas_dims["h"])

        img_rgb = render_frame_image(idx, total, grid_data, cw, ch, grid_size, mode, loop_num)

        ppm_header = f"P6\n{cw} {ch}\n255\n".encode("ascii")
        ppm_data = ppm_header + img_rgb.tobytes()
        photo = tk.PhotoImage(data=ppm_data)
        state["photo"] = photo

        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)

    def on_canvas_configure(event):
        if event.width > 20 and event.height > 20:
            if event.width != canvas_dims["w"] or event.height != canvas_dims["h"]:
                canvas_dims["w"] = event.width
                canvas_dims["h"] = event.height
                if not state["started"] or state["paused"]:
                    render_current_frame()

    canvas.bind("<Configure>", on_canvas_configure)

    btn_start = tk.Button(
        ctrl_frame, text="▶ START (Space)",
        font=("Monospace", 9, "bold"),
        bg="#28a745", fg="white", activebackground="#218838", activeforeground="white",
        relief="raised", padx=8, pady=2, cursor="hand2"
    )
    btn_start.pack(side="left", padx=2)

    btn_step_back = tk.Button(
        ctrl_frame, text="◀",
        font=("Monospace", 9),
        bg="#333", fg="white", activebackground="#555", activeforeground="white",
        relief="flat", padx=6, pady=2, cursor="hand2"
    )
    btn_step_back.pack(side="left", padx=1)

    btn_step_fwd = tk.Button(
        ctrl_frame, text="▶",
        font=("Monospace", 9),
        bg="#333", fg="white", activebackground="#555", activeforeground="white",
        relief="flat", padx=6, pady=2, cursor="hand2"
    )
    btn_step_fwd.pack(side="left", padx=1)

    btn_quit = tk.Button(
        ctrl_frame, text="✖ Quit",
        font=("Monospace", 9),
        bg="#dc3545", fg="white", activebackground="#c82333", activeforeground="white",
        relief="flat", padx=6, pady=2, cursor="hand2",
        command=lambda: root.destroy()
    )
    btn_quit.pack(side="right", padx=2)

    def update_frame():
        idx, total, grid_data = frames_data[state["idx"]]
        loop_num = (state["count"] // total_frames) + 1 if state["started"] else 1

        render_current_frame()

        pct = min(100.0, ((idx + 1) / total) * 100.0)
        is_parity = idx >= total
        tag = f"Parity {idx - total + 1}" if is_parity else f"Frame {idx + 1}/{total}"
        if not state["started"]:
            status_text = f"READY - {tag} | [SPACE] to Start"
            btn_start.config(text="▶ START (Space)", bg="#28a745", fg="white")
        elif state["paused"]:
            status_text = f"PAUSED - {tag} ({pct:5.1f}%) | [SPACE] to Resume"
            btn_start.config(text="▶ RESUME (Space)", bg="#007bff", fg="white")
        else:
            status_text = (
                f"{tag} ({pct:5.1f}%) | Loop #{loop_num} | "
                f"{1000/state['delay']:.1f} FPS"
            )
            btn_start.config(text="⏸ PAUSE (Space)", bg="#ffc107", fg="black")
        status_label.config(text=status_text)

        if state["started"] and not state["paused"]:
            state["count"] += 1
            state["idx"] = (state["idx"] + 1) % total_frames
            state["after_id"] = root.after(state["delay"], update_frame)

    def start_or_toggle(event=None):
        if not state["started"]:
            state["started"] = True
            state["paused"] = False
            update_frame()
        else:
            state["paused"] = not state["paused"]
            if state["paused"]:
                if state["after_id"]:
                    root.after_cancel(state["after_id"])
                    state["after_id"] = None
                update_frame()
            else:
                update_frame()

    def step_forward(event=None):
        if not state["started"] or state["paused"]:
            state["idx"] = (state["idx"] + 1) % total_frames
            update_frame()

    def step_backward(event=None):
        if not state["started"] or state["paused"]:
            state["idx"] = (state["idx"] - 1 + total_frames) % total_frames
            update_frame()

    def speed_up(event=None):
        state["delay"] = max(20, state["delay"] - 10)

    def slow_down(event=None):
        state["delay"] = min(300, state["delay"] + 10)

    def quit_app(event=None):
        root.destroy()

    btn_start.config(command=start_or_toggle)
    btn_step_back.config(command=step_backward)
    btn_step_fwd.config(command=step_forward)

    root.bind("<space>", start_or_toggle)
    root.bind("<Return>", start_or_toggle)
    root.bind("<Right>", step_forward)
    root.bind("<Left>", step_backward)
    root.bind("<Up>", speed_up)
    root.bind("<Down>", slow_down)
    root.bind("q", quit_app)
    root.bind("<Escape>", quit_app)

    update_frame()
    root.mainloop()

# =====================================================================
# HEADLESS SHELL TRANSFORMATION & RENDERING (--tty)
# =====================================================================

def build_terminal_frame_lines(idx, total_chunks, grid_data, grid_size, mode, loop_num, status_msg=""):
    """Renders the frame into a list of terminal text lines using ANSI escapes."""
    palette_blocks = ANSI_RGB_BLOCKS if mode == "rgb" else ANSI_BW_BLOCKS

    M = max(2, int(round(grid_size / 21.0)))
    D = grid_size + 2 * M
    total_units_w = D + 5
    total_lines = D + 5

    buffer = [[ANSI_BLACK_BLOCK for _ in range(total_units_w)] for _ in range(total_lines)]

    # 1. Corner Fiducials: 3x3 units centered at:
    # TL: (2, 2)
    # TR: (2, 2 + D)
    # BL: (2 + D, 2)
    # BR: (2 + D, 2 + D)
    fids = [
        (2, 2, ANSI_RED_BLOCK),
        (2, 2 + D, ANSI_GREEN_BLOCK),
        (2 + D, 2, ANSI_BLUE_BLOCK),
        (2 + D, 2 + D, ANSI_MAGENTA_BLOCK)
    ]
    for cr, cc, col_blk in fids:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                buffer[cr + dr][cc + dc] = ANSI_WHITE_BLOCK
        buffer[cr][cc] = col_blk

    # 2. Data Grid (starts at row 2 + M, col 2 + M)
    r_start = 2 + M
    c_start = 2 + M
    for r in range(grid_size):
        row_vals = grid_data[r]
        for c in range(grid_size):
            buffer[r_start + r][c_start + c] = palette_blocks[row_vals[c]]

    # 3. Text output lines
    out_lines = []
    pct = min(100.0, ((idx + 1) / total_chunks) * 100.0)
    is_parity = idx >= total_chunks
    tag = f"PARITY {idx - total_chunks + 1:2d}" if is_parity else f"FRAME {idx + 1:3d}/{total_chunks:3d}"
    status_header = (
        f" \033[1;33m[{tag} ({pct:5.1f}%) | "
        f"Loop #{loop_num} | {status_msg}]\033[0m"
    )
    out_lines.append(status_header)

    for row_blocks in buffer:
        out_lines.append("".join(row_blocks) + ANSI_RESET)

    bar_width = total_units_w * 2 - 12
    filled_len = min(bar_width, int(((idx + 1) / total_chunks) * bar_width))
    bar_str = ("█" if not is_parity else "▓") * filled_len + "░" * (bar_width - filled_len)
    bar_col = "\033[1;35m" if is_parity else "\033[1;32m"
    progress_line = f" {bar_col}[{bar_str}] {pct:5.1f}%\033[0m"
    out_lines.append(progress_line)

    controls_line = (
        " \033[1;36m[Space/Enter]\033[0m Start/Pause | "
        "\033[1;36m[◀/▶]\033[0m Step | "
        "\033[1;36m[▲/▼]\033[0m Speed | "
        "\033[1;31m[q]\033[0m Quit"
    )
    out_lines.append(controls_line)

    return out_lines

def get_key_nonblocking():
    """Reads a single keypress if available without blocking."""
    if select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if ch == "\033":
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        ch3 = sys.stdin.read(1)
                        if ch3 == "A": return "UP"
                        if ch3 == "B": return "DOWN"
                        if ch3 == "C": return "RIGHT"
                        if ch3 == "D": return "LEFT"
            return "ESC"
        return ch
    return None

def run_shell_transmitter(filepath, grid_size=48, delay_ms=80, mode="rgb", fec_pct=15):
    """Runs the visual transmitter directly in the shell terminal (no GUI needed)."""
    import termios
    import tty

    full_stream, orig_size, comp_size, orig_sha = prepare_stream(filepath)
    frames_data, payload_cap = pack_frames(full_stream, grid_size, mode, fec_pct=fec_pct)
    total_frames = len(frames_data)

    M = max(2, int(round(grid_size / 21.0)))
    D = grid_size + 2 * M
    total_units_w = D + 5
    required_cols = total_units_w * 2
    required_rows = total_units_w + 4

    term_size = os.get_terminal_size()
    if term_size.columns < required_cols or term_size.lines < required_rows:
        print(f"Warning: Terminal size ({term_size.columns}x{term_size.lines}) is smaller than recommended ({required_cols}x{required_rows}).")
        print("Please maximize your terminal window or reduce grid size (e.g. --grid 32).")
        try:
            input("Press [Enter] to proceed anyway, or Ctrl+C to abort...")
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

    fps = 1000.0 / delay_ms
    kb_per_sec = (payload_cap * fps) / 1024.0

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    state = {
        "idx": 0,
        "count": 0,
        "started": False,
        "paused": False,
        "delay": delay_ms
    }

    try:
        tty.setcbreak(fd)
        sys.stdout.write("\033[?1049h\033[?25l\033[2J")
        sys.stdout.flush()

        last_draw_time = 0

        while True:
            key = get_key_nonblocking()
            if key in ("q", "Q", "ESC"):
                break
            elif key in (" ", "\n", "\r"):
                if not state["started"]:
                    state["started"] = True
                    state["paused"] = False
                else:
                    state["paused"] = not state["paused"]
            elif key == "RIGHT":
                if not state["started"] or state["paused"]:
                    state["idx"] = (state["idx"] + 1) % total_frames
            elif key == "LEFT":
                if not state["started"] or state["paused"]:
                    state["idx"] = (state["idx"] - 1 + total_frames) % total_frames
            elif key == "UP":
                state["delay"] = max(20, state["delay"] - 10)
            elif key == "DOWN":
                state["delay"] = min(400, state["delay"] + 10)

            current_fps = 1000.0 / state["delay"]
            if not state["started"]:
                status_msg = "READY - Press [SPACE] / [ENTER] to Start"
            elif state["paused"]:
                status_msg = f"PAUSED - Press [SPACE] to Resume | {current_fps:.1f} FPS"
            else:
                status_msg = f"BROADCASTING @ {current_fps:.1f} FPS (~{kb_per_sec:.1f} KB/s)"

            now = time.time()
            time_since_last = (now - last_draw_time) * 1000.0

            should_update = False
            if not state["started"] or state["paused"]:
                should_update = (key is not None) or (last_draw_time == 0)
            else:
                should_update = time_since_last >= state["delay"]

            if should_update:
                idx, total, grid_data = frames_data[state["idx"]]
                loop_num = (state["count"] // total_frames) + 1 if state["started"] else 1

                lines = build_terminal_frame_lines(idx, total, grid_data, grid_size, mode, loop_num, status_msg)

                frame_output = "\033[H" + "\n".join(lines) + "\n"
                sys.stdout.write(frame_output)
                sys.stdout.flush()

                last_draw_time = now

                if state["started"] and not state["paused"]:
                    state["count"] += 1
                    state["idx"] = (state["idx"] + 1) % total_frames

            time.sleep(0.005)

    finally:
        sys.stdout.write("\033[?25h\033[?1049l\033[0m")
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\nTransmitter closed.")

# =====================================================================
# MAIN CLI ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="otd (Optical Data Transfer) v2 High-Speed Optical Transmitter")
    parser.add_argument("file", help="Path to file to transmit")
    parser.add_argument("--tty", action="store_true",
                        help="Run directly in shell/terminal (headless mode, no GUI window required)")
    parser.add_argument("--grid", "-g", type=int, default=None, choices=[32, 48, 64, 80, 96, 128],
                        help="Grid dimension (default: 64 for GUI, 48 for --tty)")
    parser.add_argument("--cell-size", "-s", type=int, default=None,
                        help="Pixel size of each cell in GUI mode (default: auto)")
    parser.add_argument("--delay", "-d", type=int, default=None,
                        help="Frame delay in milliseconds (default: 70ms for GUI, 80ms for --tty)")
    parser.add_argument("--mode", "-m", default="rgb", choices=["rgb", "bw"],
                        help="Color mode: rgb (8 colors, 3 bits/cell) or bw (2 colors, 1 bit/cell)")
    parser.add_argument("--fec", type=int, default=15,
                        help="Forward Error Correction parity percentage (default: 15, 0 to disable)")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: file '{args.file}' not found.")
        sys.exit(1)

    if args.tty:
        grid = args.grid if args.grid else 48
        delay = args.delay if args.delay else 80
        run_shell_transmitter(args.file, grid_size=grid, delay_ms=delay, mode=args.mode, fec_pct=args.fec)
    else:
        grid = args.grid if args.grid else 64
        delay = args.delay if args.delay else 70
        run_gui_transmitter(args.file, grid_size=grid, cell_px=args.cell_size, delay_ms=delay, mode=args.mode, fec_pct=args.fec)
