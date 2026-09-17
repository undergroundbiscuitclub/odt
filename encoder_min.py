#!/usr/bin/env python3
"""
otd (Optical Data Transfer) - Minimal Standalone Transmitter
Zero third-party dependencies (pure Python 3 standard library only).
Optimized for copy-pasting, keyboard emulation, and air-gapped systems.
"""
import sys, os, time, zlib, struct, hashlib

PALETTE = [(0,0,0), (0,0,255), (0,255,0), (0,255,255), (255,0,0), (255,0,255), (255,255,0), (255,255,255)]
ANSI = [f"\033[48;2;{r};{g};{b}m  " for r, g, b in PALETTE]
W_BLK, K_BLK = "\033[48;2;255;255;255m  ", "\033[48;2;0;0;0m  "
FIDS = ["\033[48;2;255;0;0m  ", "\033[48;2;0;255;0m  ", "\033[48;2;0;0;255m  ", "\033[48;2;255;0;255m  "]
FID_COLS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]

_GF_EXP, _GF_LOG = [0] * 512, [0] * 256
_x = 1
for _i in range(255):
    _GF_EXP[_i] = _x
    _GF_EXP[_i + 255] = _x
    _GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100: _x ^= 0x11d
_GF_LOG[0] = 511

def _gf_mul(a, b): return 0 if a == 0 or b == 0 else _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]
def _gf_inv(a): return _GF_EXP[255 - _GF_LOG[a]]
def _cauchy_coeff(r, c): return _gf_inv((255 - r) ^ c)

def generate_fec_parities(chunks, M):
    K, max_len = len(chunks), max(len(c) for c in chunks)
    parities = []
    for r in range(M):
        p = bytearray(max_len)
        for c in range(K):
            coeff = _cauchy_coeff(r, c)
            d = chunks[c].ljust(max_len, b"\x00")
            for i in range(max_len):
                if d[i]: p[i] ^= _gf_mul(coeff, d[i])
        parities.append(bytes(p))
    return parities

def pack_file(path, gs=48, fec_pct=15):
    with open(path, "rb") as f:
        raw = f.read()
    comp = zlib.compress(raw, 9)
    bn = os.path.basename(path).encode("utf-8")
    meta = struct.pack(">4sQQ32sH", b"QSMD", len(raw), len(comp), hashlib.sha256(raw).digest(), len(bn)) + bn
    stream = meta + comp
    bpf = (gs * gs * 3) // 8
    cap = bpf - 16
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
        bits = "".join(format(b, "08b") for b in (hdr + c).ljust(bpf, b"\x00"))
        vals = [int(bits[i:i + 3], 2) for i in range(0, len(bits) - 2, 3)]
        if len(vals) < tot_cells:
            vals.extend([0] * (tot_cells - len(vals)))
        frames.append(vals[:tot_cells])
    return frames, tot, len(raw), len(comp), data_len

def render_tty(cells, gs=48):
    M = max(2, int(round(gs / 21.0)))
    D = gs + 2 * M
    w = D + 5
    buf = [[K_BLK] * w for _ in range(w)]
    for cr, cc, fid in ((2, 2, 0), (2, 2 + D, 1), (2 + D, 2, 2), (2 + D, 2 + D, 3)):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                buf[cr + dr][cc + dc] = W_BLK
        buf[cr][cc] = FIDS[fid]
    r_start = 2 + M
    c_start = 2 + M
    for r in range(gs):
        for c in range(gs):
            buf[r_start + r][c_start + c] = ANSI[cells[r * gs + c]]
    return "\033[H" + "\n".join("".join(row) + "\033[0m" for row in buf) + "\n"

def run_tty(frames, tot, delay_ms, gs=48, data_len=None):
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
    state = {"idx": 0, "run": False, "loop": 1}
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
                tag = f"Parity {state['idx'] - data_len + 1:2d}" if is_parity else f"Frame {state['idx'] + 1:3d}/{data_len:3d}"
                st = "BROADCASTING" if state["run"] else "READY (Press Space/Enter)"
                status = f"\033[1;33m[{st}] {tag} ({pct:5.1f}%) | Loop #{state['loop']} | [Space] Toggle | [q] Quit\033[0m\n"
                sys.stdout.write(render_tty(frames[state["idx"]], gs) + status)
                sys.stdout.flush()
                last_t = now

                if state["run"]:
                    state["idx"] = (state["idx"] + 1) % tot
                    if state["idx"] == 0:
                        state["loop"] += 1
            time.sleep(0.01)
    finally:
        sys.stdout.write("\033[?25h\033[?1049l\033[0m")
        sys.stdout.flush()
        if fd is not None and old_attr is not None:
            import termios
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        print("\nTransmitter closed.")

def run_gui(frames, tot, delay_ms, fname, gs=48, data_len=None):
    if data_len is None: data_len = tot
    import tkinter as tk
    init_dim = 600

    root = tk.Tk()
    root.title(f"otd Transmitter [{fname}]")
    root.configure(bg="black")
    root.resizable(True, True)
    root.minsize(150, 150)

    lbl = tk.Label(root, text="READY - Click or Press Space", fg="yellow", bg="black", font=("Monospace", 9, "bold"))
    lbl.pack(side="top", fill="x", pady=2)

    canvas = tk.Canvas(root, width=init_dim, height=init_dim, bg="black", highlightthickness=0)
    canvas.pack(side="top", fill="both", expand=True, padx=4, pady=2)

    state = {"idx": 0, "run": False, "loop": 1, "photo": None, "w": init_dim, "h": init_dim}

    def make_ppm(cells, w, h):
        buf = bytearray(w * h * 3)
        msize_w = max(4, int(w * 0.024))
        msize_h = max(4, int(h * 0.024))
        PAL_B = [bytes(c) for c in PALETTE]
        FID_B = [bytes(c) for c in FID_COLS]
        fids = (
            (int(w * 0.04), int(h * 0.04), FID_B[0]),
            (int(w * 0.96), int(h * 0.04), FID_B[1]),
            (int(w * 0.04), int(h * 0.96), FID_B[2]),
            (int(w * 0.96), int(h * 0.96), FID_B[3]),
        )
        for cx, cy, col in fids:
            for y in range(max(0, cy - msize_h), min(h, cy + msize_h)):
                row_off = y * w * 3
                for x in range(max(0, cx - msize_w), min(w, cx + msize_w)):
                    buf[row_off + x * 3 : row_off + x * 3 + 3] = b"\xff\xff\xff"
            rw, rh = max(1, msize_w // 3), max(1, msize_h // 3)
            for y in range(max(0, cy - msize_h + rh), min(h, cy + msize_h - rh)):
                row_off = y * w * 3
                for x in range(max(0, cx - msize_w + rw), min(w, cx + msize_w - rw)):
                    buf[row_off + x * 3 : row_off + x * 3 + 3] = b"\x00\x00\x00"
            cw, ch_c = max(2, msize_w * 2 // 3), max(2, msize_h * 2 // 3)
            for y in range(max(0, cy - msize_h + ch_c), min(h, cy + msize_h - ch_c)):
                row_off = y * w * 3
                for x in range(max(0, cx - msize_w + cw), min(w, cx + msize_w - cw)):
                    buf[row_off + x * 3 : row_off + x * 3 + 3] = col

        gx0 = int(w * 0.08)
        gy0 = int(h * 0.08)
        gw = max(gs, int(w * 0.84))
        gh = max(gs, int(h * 0.84))

        col_map = [c * gs // gw for c in range(gw)]
        for r in range(gh):
            cell_r = r * gs // gh
            row_offset = (gy0 + r) * w * 3
            row_cells = [PAL_B[cells[cell_r * gs + col_map[c]]] for c in range(gw)]
            buf[row_offset + gx0 * 3 : row_offset + (gx0 + gw) * 3] = b"".join(row_cells)

        return f"P6\n{w} {h}\n255\n".encode("ascii") + bytes(buf)

    def render():
        cw = max(50, state["w"])
        ch = max(50, state["h"])
        base_ppm = make_ppm(frames[state["idx"]], cw, ch)
        state["photo"] = tk.PhotoImage(data=base_ppm)
        canvas.itemconfig(img_id, image=state["photo"])
        canvas.coords(img_id, 0, 0)

    img_id = canvas.create_image(0, 0, anchor="nw")
    render()

    def on_resize(event):
        if event.width > 20 and event.height > 20:
            if event.width != state["w"] or event.height != state["h"]:
                state["w"] = event.width
                state["h"] = event.height
                render()

    canvas.bind("<Configure>", on_resize)

    def tick():
        if state["run"]:
            state["idx"] = (state["idx"] + 1) % tot
            if state["idx"] == 0:
                state["loop"] += 1
            render()
            pct = min(100.0, ((state["idx"] + 1) / data_len) * 100.0)
            tag = f"Parity {state['idx'] - data_len + 1}" if state['idx'] >= data_len else f"Frame {state['idx'] + 1}/{data_len}"
            lbl.config(text=f"BROADCASTING: {tag} ({pct:.1f}%) | Loop #{state['loop']}")
        root.after(delay_ms, tick)

    def toggle(e=None):
        state["run"] = not state["run"]
        pct = min(100.0, ((state["idx"] + 1) / data_len) * 100.0)
        tag = f"Parity {state['idx'] - data_len + 1}" if state['idx'] >= data_len else f"Frame {state['idx'] + 1}/{data_len}"
        if state["run"]:
            lbl.config(text=f"BROADCASTING: {tag} ({pct:.1f}%)")
        else:
            lbl.config(text="PAUSED - Press Space to Resume")

    root.bind("<space>", toggle)
    root.bind("<Return>", toggle)
    root.bind("<Button-1>", toggle)
    root.bind("q", lambda e: root.destroy())
    root.bind("<Escape>", lambda e: root.destroy())
    root.after(delay_ms, tick)
    root.mainloop()

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("otd minimal transmitter (zero dependencies)")
        print("Usage: python3 encoder_min.py <file> [--tty] [--gui] [--delay MS] [--grid N] [--fec PCT]")
        sys.exit(0)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Error: file '{path}' not found.")
        sys.exit(1)

    delay = 80
    gs = 48
    fec = 15
    force_tty = "--tty" in sys.argv
    force_gui = "--gui" in sys.argv

    for i, a in enumerate(sys.argv):
        if a == "--delay" and i + 1 < len(sys.argv): delay = int(sys.argv[i + 1])
        if a == "--grid" and i + 1 < len(sys.argv): gs = int(sys.argv[i + 1])
        if a == "--fec" and i + 1 < len(sys.argv): fec = int(sys.argv[i + 1])

    frames, tot, osize, csize, data_len = pack_file(path, gs=gs, fec_pct=fec)
    parity_count = tot - data_len
    fec_str = f" (+{parity_count} FEC parity)" if parity_count > 0 else ""
    print(f"File: {os.path.basename(path)} | Original: {osize:,} B | Compressed: {csize:,} B | Frames: {tot}{fec_str}")

    if not force_tty:
        try:
            run_gui(frames, tot, delay, os.path.basename(path), gs=gs, data_len=data_len)
            return
        except Exception as e:
            if force_gui:
                print(f"GUI failed: {e}")
                sys.exit(1)

    run_tty(frames, tot, delay, gs=gs, data_len=data_len)

if __name__ == "__main__":
    main()
