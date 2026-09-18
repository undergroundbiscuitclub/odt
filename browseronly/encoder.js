/**
 * browseronly/encoder.js - In-Browser Optical Screen Transmitter Engine
 * Encodes files into Q2 optical frames with Cauchy Reed-Solomon FEC,
 * CRC32 verification, and renders at 10-60 FPS on HTML5 Canvas.
 */

(function(exports) {
  'use strict';

  const PALETTE_RGB = [
    [0, 0, 0],       // 000: Black
    [0, 0, 255],     // 001: Blue
    [0, 255, 0],     // 010: Green
    [0, 255, 255],   // 011: Cyan
    [255, 0, 0],     // 100: Red
    [255, 0, 255],   // 101: Magenta
    [255, 255, 0],   // 110: Yellow
    [255, 255, 255]  // 111: White
  ];

  const PALETTE_CSS = [
    "#000000",
    "#0000ff",
    "#00ff00",
    "#00ffff",
    "#ff0000",
    "#ff00ff",
    "#ffff00",
    "#ffffff"
  ];

  // Helper: deflate compress using native browser CompressionStream or fallback
  async function compressDeflate(rawBytes) {
    if (typeof CompressionStream !== "undefined") {
      try {
        const cs = new CompressionStream("deflate");
        const writer = cs.writable.getWriter();
        writer.write(rawBytes);
        writer.close();
        const response = new Response(cs.readable);
        const buffer = await response.arrayBuffer();
        return new Uint8Array(buffer);
      } catch (e) {
        console.warn("Native CompressionStream failed, falling back:", e);
      }
    }
    // Fallback: uncompressed / raw if no compressor
    return rawBytes;
  }

  // Pack file into visual frames
  async function packFileToFrames(filename, rawBytes, gridSize = 64, fecPct = 15) {
    const compBytes = await compressDeflate(rawBytes);

    // SHA-256 calculation
    const hashBuf = await crypto.subtle.digest("SHA-256", rawBytes);
    const shaBytes = new Uint8Array(hashBuf);
    let shaHex = "";
    for (let i = 0; i < 32; i++) shaHex += shaBytes[i].toString(16).padStart(2, "0");

    // Metadata header (54 bytes minimum + filename)
    const enc = new TextEncoder();
    const fnameBytes = enc.encode(filename);
    const metaLen = 54 + fnameBytes.length;
    const metaBuf = new Uint8Array(metaLen);
    const view = new DataView(metaBuf.buffer);

    // b"QSMD"
    metaBuf[0] = 0x51; metaBuf[1] = 0x53; metaBuf[2] = 0x4d; metaBuf[3] = 0x44;
    view.setBigUint64(4, BigInt(rawBytes.length), false);
    view.setBigUint64(12, BigInt(compBytes.length), false);
    metaBuf.set(shaBytes, 20);
    view.setUint16(52, fnameBytes.length, false);
    metaBuf.set(fnameBytes, 54);

    // Concatenate metadata + compressed stream
    const fullStream = new Uint8Array(metaLen + compBytes.length);
    fullStream.set(metaBuf, 0);
    fullStream.set(compBytes, metaLen);

    // Chunk size: bytes per frame = (gridSize * gridSize * 3) / 8
    const bpf = Math.floor((gridSize * gridSize * 3) / 8);
    const cap = bpf - 16; // 16 bytes Q2 header

    const chunks = [];
    for (let i = 0; i < fullStream.length; i += cap) {
      chunks.push(fullStream.subarray(i, Math.min(fullStream.length, i + cap)));
    }
    const dataLen = chunks.length;
    const allChunks = [...chunks];

    // Cauchy Reed-Solomon FEC Parity generation
    if (fecPct > 0 && dataLen > 0 && window.OTDFec && window.OTDFec.generateFecParities) {
      let mCount = Math.max(2, Math.round(dataLen * (fecPct / 100.0)));
      mCount = Math.min(mCount, Math.max(0, 255 - dataLen));
      if (mCount > 0) {
        const parities = window.OTDFec.generateFecParities(chunks, mCount);
        allChunks.push(...parities);
      }
    }

    const totFrames = allChunks.length;
    const totCells = gridSize * gridSize;
    const frames = [];

    for (let idx = 0; idx < totFrames; idx++) {
      const c = allChunks[idx];
      const isParity = idx >= dataLen;
      const crc = window.OTDDecoder ? window.OTDDecoder.computeCRC32(c) : 0;
      const flags = (idx === 0 ? 1 : 0) | (isParity ? 2 : 0);

      // 16 bytes Q2 header
      const hdr = new Uint8Array(16);
      const hView = new DataView(hdr.buffer);
      hdr[0] = 0x51; hdr[1] = 0x32; // 'Q', '2'
      hdr[2] = 2;                  // Version 2
      hdr[3] = 3;                  // RGB 8-color mode
      hdr[4] = gridSize;
      hdr[5] = flags;
      hView.setUint16(6, idx, false);
      hView.setUint16(8, dataLen, false);
      hView.setUint16(10, c.length, false);
      hView.setUint32(12, crc, false);

      // Pack payload bytes into stream
      const frameBytes = new Uint8Array(bpf);
      frameBytes.set(hdr, 0);
      frameBytes.set(c, 16);

      // Convert bytes to 3-bit color cells
      const cellVals = new Uint8Array(totCells);
      let bitPos = 0;
      for (let i = 0; i < totCells; i++) {
        const byteIdx = Math.floor(bitPos / 8);
        const bitOffset = bitPos % 8;
        let val = 0;

        if (bitOffset <= 5) {
          val = (frameBytes[byteIdx] >> (5 - bitOffset)) & 7;
        } else if (bitOffset === 6) {
          val = ((frameBytes[byteIdx] & 3) << 1) | ((frameBytes[byteIdx + 1] >> 7) & 1);
        } else if (bitOffset === 7) {
          val = ((frameBytes[byteIdx] & 1) << 2) | ((frameBytes[byteIdx + 1] >> 6) & 3);
        }
        cellVals[i] = val & 7;
        bitPos += 3;
      }

      frames.push({
        idx,
        total: totFrames,
        dataLen,
        isParity,
        cells: cellVals
      });
    }

    return {
      frames,
      totalFrames: totFrames,
      dataFrames: dataLen,
      origSize: rawBytes.length,
      compSize: compBytes.length,
      sha256: shaHex
    };
  }

  // Draw a frame onto canvas with fiducials & grid
  function drawOpticalFrame(canvas, cells, gridSize) {
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const scale = W / 1000.0;

    // Background Black
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, W, H);

    // Draw Corner Fiducials
    // Canonical centers: TL (40, 40), TR (960, 40), BR (960, 960), BL (40, 960)
    const fidDefs = [
      { cx: 40, cy: 40, col: "#ff0000" },   // TL Red
      { cx: 960, cy: 40, col: "#00ff00" },  // TR Green
      { cx: 960, cy: 960, col: "#ff00ff" }, // BR Magenta
      { cx: 40, cy: 960, col: "#0000ff" }   // BL Blue
    ];

    const outerRad = 22.0 * scale;
    const innerRad = 15.0 * scale;

    for (const f of fidDefs) {
      const px = f.cx * scale;
      const py = f.cy * scale;

      // Outer White ring
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(px - outerRad, py - outerRad, outerRad * 2, outerRad * 2);

      // Core chromatic square
      ctx.fillStyle = f.col;
      ctx.fillRect(px - innerRad, py - innerRad, innerRad * 2, innerRad * 2);
    }

    // White Outer Margins Reference lines (outer borders)
    ctx.fillStyle = "#ffffff";
    // Top border between 36..44 at Y=21
    ctx.fillRect((40 - 5) * scale, 18 * scale, 10 * scale, 7 * scale);
    ctx.fillRect((960 - 5) * scale, 18 * scale, 10 * scale, 7 * scale);

    // Data Grid: GUI geometry spans 80 to 920 (span 840)
    const gStart = 80.0 * scale;
    const gSpan = 840.0 * scale;
    const cellStep = gSpan / gridSize;

    for (let r = 0; r < gridSize; r++) {
      const y = gStart + r * cellStep;
      for (let c = 0; c < gridSize; c++) {
        const x = gStart + c * cellStep;
        const colIdx = cells[r * gridSize + c];
        ctx.fillStyle = PALETTE_CSS[colIdx];
        ctx.fillRect(x, y, cellStep + 0.3, cellStep + 0.3); // slight overlap avoids seam artifacts
      }
    }
  }

  // Broadcaster Animation Controller
  class BrowserOpticalBroadcaster {
    constructor(canvas) {
      this.canvas = canvas;
      this.frames = [];
      this.totalFrames = 0;
      this.currentIdx = 0;
      this.gridSize = 64;
      this.fps = 30;
      this.isPlaying = false;
      this.animId = null;
      this.lastTime = 0;
      this.onProgress = null;
    }

    async stageFile(filename, rawBytes, gridSize = 64, fecPct = 15) {
      this.gridSize = gridSize;
      const res = await packFileToFrames(filename, rawBytes, gridSize, fecPct);
      this.frames = res.frames;
      this.totalFrames = res.totalFrames;
      this.currentIdx = 0;
      this.renderCurrent();
      return res;
    }

    renderCurrent() {
      if (this.frames.length > 0 && this.canvas) {
        const f = this.frames[this.currentIdx];
        drawOpticalFrame(this.canvas, f.cells, this.gridSize);
        if (this.onProgress) {
          this.onProgress(this.currentIdx, this.totalFrames, f.isParity);
        }
      }
    }

    start(fps = 30) {
      this.fps = fps;
      this.isPlaying = true;
      this.lastTime = performance.now();
      const interval = 1000.0 / this.fps;

      const loop = (now) => {
        if (!this.isPlaying) return;
        const delta = now - this.lastTime;
        if (delta >= interval) {
          this.currentIdx = (this.currentIdx + 1) % this.totalFrames;
          this.renderCurrent();
          this.lastTime = now - (delta % interval);
        }
        this.animId = requestAnimationFrame(loop);
      };

      this.animId = requestAnimationFrame(loop);
    }

    stop() {
      this.isPlaying = false;
      if (this.animId) {
        cancelAnimationFrame(this.animId);
        this.animId = null;
      }
    }

    seek(idx) {
      if (idx >= 0 && idx < this.totalFrames) {
        this.currentIdx = idx;
        this.renderCurrent();
      }
    }

    setFps(fps) {
      this.fps = Math.max(1, Math.min(60, fps));
    }
  }

  exports.PALETTE_RGB = PALETTE_RGB;
  exports.PALETTE_CSS = PALETTE_CSS;
  exports.packFileToFrames = packFileToFrames;
  exports.drawOpticalFrame = drawOpticalFrame;
  exports.BrowserOpticalBroadcaster = BrowserOpticalBroadcaster;

})(typeof module !== 'undefined' && module.exports ? module.exports : (window.OTDEncoder = {}));
