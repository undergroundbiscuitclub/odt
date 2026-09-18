#!/usr/bin/env python3
"""
browseronly/run.py - Air-Gap Duplex Web Server with In-Browser Computer Vision
Hosts the client-side optical decoder & encoder UI and provides backend
keyboard typing (ServerDuplexSessionV2) and file management.

Usage:
    python3 browseronly/run.py [port]
"""

import argparse
import base64
import http.server
import json
import os
import socketserver
import sys
import time
import urllib.parse
import webbrowser

DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import ServerDuplexSessionV2 from server_v2 and format_bytes
try:
    from server_v2 import ServerDuplexSessionV2, format_bytes
except ImportError:
    from server import ServerDuplexSession
    ServerDuplexSessionV2 = ServerDuplexSession
    def format_bytes(n):
        if n is None: return "0 B"
        for unit in ["B", "KB", "MB", "GB"]:
            if abs(n) < 1024.0: return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
            n /= 1024.0

class BrowserDuplexSession(ServerDuplexSessionV2):
    """
    Enhanced Duplex Session optimized for in-browser client-side optical feedback:
    - Adaptive chunk timeouts based on packet length and keyboard speed preset
    - Resilient optical link dead-man's switch timeout (3.5s)
    - Heartbeat-aware transmission loop preventing premature aborts on large files
    """
    def is_link_alive(self, timeout=3.5):
        with self.lock:
            if not self.transfer_active or self.abort_requested:
                return False
            if not self.optical_locked:
                return False
            if time.time() - self.last_optical_decode_time > timeout:
                return False
            if self.client_state == 4:  # STATE_ERROR
                return False
            return True

    def _transmission_worker_loop(self):
        with self.lock:
            if not self.staged_file or self.transfer_active or self.transfer_complete:
                return
            self.transfer_active = True
            self.abort_requested = False
            self.start_time = time.time()
            staged = self.staged_file

        self.log(f"Starting keyboard transmission of '{staged['filename']}' ({staged['total_chunks']} chunks)...")
        cancel_header_cb = lambda: not self.is_link_alive(timeout=4.0)
        cancel_data_cb = lambda: not self.is_link_alive(timeout=3.5)

        try:
            time.sleep(0.15)

            # 1. Send Header packet
            header_pkt = staged["header_pkt"]
            header_acked = False
            for attempt in range(8):
                if not self.is_link_alive(timeout=4.0):
                    self.abort_transmission("Optical link lost before header acknowledged")
                    return

                self.log(f"Sending header packet (attempt {attempt + 1}/8)...")
                ok = self.typer.type_string(header_pkt, delay=self.char_delay, cancel_check=cancel_header_cb)
                if not ok or not self.is_link_alive(timeout=4.0):
                    self.abort_transmission("Optical link interrupted during header typing")
                    return

                t_wait = time.time() + 2.0
                while time.time() < t_wait:
                    time.sleep(0.04)
                    if not self.is_link_alive(timeout=4.0):
                        self.abort_transmission("Optical link lost while awaiting header ACK")
                        return
                    with self.lock:
                        if self.client_state == 2:  # STATE_RECEIVING
                            header_acked = True
                            break
                if header_acked:
                    self.log("Client acknowledged header packet on optical grid (STATE_RECEIVING)!")
                    break

            if not header_acked:
                self.abort_transmission("Client did not acknowledge header on optical grid after 8 attempts. Halting to avoid typing into wrong window.")
                return

            # 2. Transmit Data chunks with Lock-Step Optical Acknowledgment
            tot = staged["total_chunks"]
            max_chunk_retries = 10

            for idx in range(tot):
                if self.is_chunk_acked(idx):
                    continue

                chunk_acked = False
                for attempt in range(max_chunk_retries):
                    if not self.is_link_alive(timeout=3.5):
                        self.abort_transmission(f"Optical link lost at chunk {idx + 1}/{tot}")
                        return

                    if attempt > 0:
                        self.typer.type_string("\n")
                        time.sleep(0.02)

                    pkt = staged["data_pkts"][idx]
                    ok = self.typer.type_string(pkt, delay=self.char_delay, cancel_check=cancel_data_cb)
                    if not ok or not self.is_link_alive(timeout=3.5):
                        self.abort_transmission(f"Optical link interrupted during chunk {idx + 1}/{tot}")
                        return

                    with self.lock:
                        self.chunks_sent += 1

                    # Adaptive timeout: allow adequate time for typing + optical roundtrip
                    est_typing_time = len(pkt) * self.char_delay
                    t_chunk_wait = time.time() + max(1.5, 0.8 + est_typing_time)
                    while time.time() < t_chunk_wait:
                        time.sleep(0.01)
                        if not self.is_link_alive(timeout=3.5):
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
                        self.log(f"Chunk {idx + 1}/{tot} not confirmed within timeout, retrying (attempt {attempt + 2}/{max_chunk_retries})...")

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
                self.abort_transmission(f"Missing {len(missing)} chunks after transmission: {missing[:5]}...")
                return

            self.log("All chunks sent and optically acknowledged. Sending end-of-stream packet...")
            end_pkt = staged["end_pkt"]
            cancel_end_cb = lambda: not self.is_link_alive(timeout=4.0)

            for attempt in range(8):
                if not self.is_link_alive(timeout=4.0):
                    self.abort_transmission("Optical link lost before sending end packet")
                    return

                self.typer.type_string(end_pkt, delay=self.char_delay, cancel_check=cancel_end_cb)
                if not self.is_link_alive(timeout=4.0):
                    self.abort_transmission("Optical link lost during end-of-stream")
                    return

                t_end_wait = time.time() + 1.5
                while time.time() < t_end_wait:
                    time.sleep(0.04)
                    if not self.is_link_alive(timeout=4.0):
                        self.abort_transmission("Optical link lost during final verification")
                        return
                    with self.lock:
                        if self.client_state == 3:  # STATE_COMPLETE
                            self.transfer_active = False
                            self.transfer_complete = True
                            self.end_time = time.time()
                            self.log("Transmission 100% COMPLETE & VERIFIED by client optical grid!")
                            self.typer.release_all_keys()
                            return

            with self.lock:
                if self.client_state == 3:
                    self.transfer_active = False
                    self.transfer_complete = True
                    self.end_time = time.time()
                    self.log("Transmission 100% COMPLETE & VERIFIED by client optical grid!")
                else:
                    self.log("[Warning] End packet sent but client has not yet signaled STATE_COMPLETE.")
        except Exception as e:
            self.abort_transmission(f"Transmission error: {e}")


PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8088
OUTPUT_DIR = os.path.abspath(os.path.join(PARENT_DIR, "received"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shared duplex session for keyboard sending
session = BrowserDuplexSession(chunk_size=256, char_delay=0.0002)


def get_received_files_list():
    files = []
    if not os.path.exists(OUTPUT_DIR):
        return files
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath) and not fname.startswith("."):
            try:
                st = os.stat(fpath)
                mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
                files.append({
                    "filename": fname,
                    "size": st.st_size,
                    "size_str": format_bytes(st.st_size),
                    "mtime": st.st_mtime,
                    "mtime_str": mtime_str
                })
            except Exception:
                pass
    return sorted(files, key=lambda x: x["mtime"], reverse=True)


class DuplexServerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def log_message(self, format, *args):
        # Suppress routine 200 GET logs
        if len(args) > 1 and "200" not in str(args[1]):
            super().log_message(format, *args)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_file_response(os.path.join(DIR, "index.html"), "text/html; charset=utf-8")
        elif path == "/api/status":
            self.send_json_response(200, {
                "send": session.get_telemetry_dict(),
                "files": get_received_files_list(),
                "backend": session.typer.backend,
                "speed_preset": session.speed_preset
            })
        elif path == "/api/files":
            self.send_json_response(200, get_received_files_list())
        elif path.startswith("/download/"):
            raw_name = path[len("/download/"):]
            filename = os.path.basename(urllib.parse.unquote(raw_name))
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(file_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(os.path.getsize(file_path)))
                self.end_headers()
                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk: break
                        self.wfile.write(chunk)
            else:
                self.send_error(404, "File not found")
        else:
            # Fallback to serving static files from DIR
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""

        if path == "/api/upload":
            ctype = self.headers.get("Content-Type", "")
            filename = "transfer_file.bin"
            raw_bytes = body
            if "multipart/form-data" in ctype:
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
            self.send_json_response(200 if ok else 400, {
                "status": "ok" if ok else "error",
                "message": msg,
                "send": session.get_telemetry_dict()
            })

        elif path == "/api/stage_received":
            try:
                data = json.loads(body.decode("utf-8"))
                filename = os.path.basename(data.get("filename", ""))
                fpath = os.path.join(OUTPUT_DIR, filename)
                if not os.path.isfile(fpath):
                    self.send_json_response(404, {"status": "error", "message": "File not found"})
                    return
                with open(fpath, "rb") as f:
                    raw_bytes = f.read()
                ok, msg = session.stage_file(filename, raw_bytes)
                self.send_json_response(200 if ok else 400, {
                    "status": "ok" if ok else "error",
                    "message": msg,
                    "send": session.get_telemetry_dict()
                })
            except Exception as e:
                self.send_json_response(400, {"status": "error", "message": str(e)})

        elif path == "/api/speed":
            try:
                data = json.loads(body.decode("utf-8"))
                preset = data.get("preset", "fast")
                session.set_speed_preset(preset)
                self.send_json_response(200, {
                    "status": "ok",
                    "preset": preset,
                    "send": session.get_telemetry_dict()
                })
            except Exception as e:
                self.send_json_response(400, {"status": "error", "message": str(e)})

        elif path == "/api/chat":
            try:
                data = json.loads(body.decode("utf-8"))
                text = data.get("text", "")
                if text:
                    session.send_chat(text)
                self.send_json_response(200, {"status": "ok"})
            except Exception as e:
                self.send_json_response(400, {"status": "error", "message": str(e)})

        elif path == "/api/abort":
            session.abort_transmission("User or browser requested abort")
            self.send_json_response(200, {
                "status": "ok",
                "message": "Transmission halted",
                "send": session.get_telemetry_dict()
            })

        elif path == "/api/request_keyboard":
            ok = session.typer.request_access()
            self.send_json_response(200, {"status": "ok", "granted": ok})

        elif path == "/api/ack":
            # Browser sends optical ACK detected by local client-side computer vision
            try:
                data = json.loads(body.decode("utf-8"))
                b64_payload = data.get("payload_b64")
                fiducials = data.get("fiducials")
                if b64_payload:
                    payload = base64.b64decode(b64_payload)
                    res = (0, 1, payload, 32, "rgb")
                    telemetry = session.update_with_decoded_ack(res, fiducials)
                else:
                    telemetry = session.update_with_decoded_ack(None, fiducials)
                self.send_json_response(200, {"status": "ok", "send": telemetry})
            except Exception as e:
                self.send_json_response(400, {"status": "error", "message": str(e)})

        elif path == "/api/save_file":
            # Saves a file received and reassembled by browser into ./received
            try:
                data = json.loads(body.decode("utf-8"))
                fname = os.path.basename(data.get("filename", "received_file.bin"))
                b64_data = data.get("data_b64", "")
                raw_bytes = base64.b64decode(b64_data)
                fpath = os.path.join(OUTPUT_DIR, fname)
                with open(fpath, "wb") as f:
                    f.write(raw_bytes)
                self.send_json_response(200, {
                    "status": "ok",
                    "filename": fname,
                    "size": len(raw_bytes),
                    "files": get_received_files_list()
                })
            except Exception as e:
                self.send_json_response(400, {"status": "error", "message": str(e)})

        elif path == "/api/delete_file":
            try:
                data = json.loads(body.decode("utf-8"))
                fname = os.path.basename(data.get("filename", ""))
                fpath = os.path.join(OUTPUT_DIR, fname)
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    self.send_json_response(200, {"status": "ok", "files": get_received_files_list()})
                else:
                    self.send_json_response(404, {"status": "error", "message": "File not found"})
            except Exception as e:
                self.send_json_response(400, {"status": "error", "message": str(e)})

        else:
            self.send_json_response(404, {"status": "error", "message": "Not found"})

    def send_json_response(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_file_response(self, filepath, content_type):
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(filepath, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk: break
                    self.wfile.write(chunk)
        else:
            self.send_error(404, "File not found")


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    target_port = PORT
    httpd = None
    for p in range(target_port, target_port + 10):
        try:
            httpd = ThreadedTCPServer(("", p), DuplexServerHandler)
            target_port = p
            break
        except OSError:
            continue

    if not httpd:
        print(f"Error: Could not bind to port {PORT} or nearby ports.")
        sys.exit(1)

    url = f"http://localhost:{target_port}/index.html"
    print("=" * 68)
    print("  ⚡ OTD IN-BROWSER DUPLEX AIR-GAP SERVER (Send & Receive)")
    print("=" * 68)
    print(f"  URL:              {url}")
    print(f"  Static Root:      {DIR}")
    print(f"  Received Files:   {OUTPUT_DIR}")
    print(f"  Keyboard Backend: {session.typer.backend.upper()}")
    print("  Optical Engine:   100% Client-Side WebGL / Canvas Vision (<1ms lag)")
    print("=" * 68)
    print("Press Ctrl+C to stop server.\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server stopped]")


if __name__ == "__main__":
    main()
