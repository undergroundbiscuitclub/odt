/**
 * browseronly/inflate.js - Fast In-Browser Zlib / Deflate Decompressor
 * Uses native Web Streams DecompressionStream with a zero-dependency pure JS fallback.
 */

(function(exports) {
  'use strict';

  // Minimal RFC 1951 Deflate decompressor fallback
  function puffInflate(source) {
    // Strip zlib header (RFC 1950) if present: 2 bytes header + 4 bytes adler32 trailer
    let input = source;
    if (source.length > 6 && (source[0] & 0x0f) === 8 && (((source[0] << 8) | source[1]) % 31 === 0)) {
      input = source.subarray(2, source.length - 4);
    }

    let bitBuf = 0, bitCount = 0, inPos = 0;
    function needBits(n) {
      while (bitCount < n) {
        if (inPos >= input.length) return false;
        bitBuf |= input[inPos++] << bitCount;
        bitCount += 8;
      }
      return true;
    }
    function getBits(n) {
      needBits(n);
      const val = bitBuf & ((1 << n) - 1);
      bitBuf >>>= n;
      bitCount -= n;
      return val;
    }

    const outChunks = [];
    let outChunk = new Uint8Array(65536);
    let outPos = 0;

    function putByte(b) {
      if (outPos >= outChunk.length) {
        outChunks.push(outChunk);
        outChunk = new Uint8Array(65536);
        outPos = 0;
      }
      outChunk[outPos++] = b;
    }

    // Fixed Huffman trees (RFC 1951 §3.2.6)
    const fixedLitTable = new Int16Array(288);
    for (let i = 0; i <= 143; i++) fixedLitTable[i] = 8;
    for (let i = 144; i <= 255; i++) fixedLitTable[i] = 9;
    for (let i = 256; i <= 279; i++) fixedLitTable[i] = 7;
    for (let i = 280; i <= 287; i++) fixedLitTable[i] = 8;

    function buildHuffman(lengths, maxBits) {
      const count = new Uint16Array(maxBits + 1);
      for (let i = 0; i < lengths.length; i++) {
        if (lengths[i] > 0) count[lengths[i]]++;
      }
      const nextCode = new Uint32Array(maxBits + 1);
      let code = 0;
      for (let bits = 1; bits <= maxBits; bits++) {
        code = (code + count[bits - 1]) << 1;
        nextCode[bits] = code;
      }
      const table = {};
      for (let i = 0; i < lengths.length; i++) {
        const len = lengths[i];
        if (len > 0) {
          const c = nextCode[len]++;
          // Reverse bits of length `len`
          let rev = 0;
          for (let j = 0; j < len; j++) rev = (rev << 1) | ((c >>> j) & 1);
          table[(len << 16) | rev] = i;
        }
      }
      return table;
    }

    const fixedLitTree = buildHuffman(fixedLitTable, 9);
    const fixedDistLengths = new Uint8Array(32);
    fixedDistLengths.fill(5);
    const fixedDistTree = buildHuffman(fixedDistLengths, 5);

    function decodeSymbol(tree, maxBits) {
      let code = 0;
      for (let len = 1; len <= maxBits; len++) {
        code = (code | (getBits(1) << (len - 1)));
        const key = (len << 16) | code;
        if (key in tree) return tree[key];
      }
      return -1;
    }

    const LENS = [3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258];
    const LEXT = [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0];
    const DISTS = [1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577];
    const DEXT = [0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13];
    const CODE_ORDER = [16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15];

    let lastBlock = false;
    while (!lastBlock) {
      lastBlock = getBits(1) === 1;
      const btype = getBits(2);
      if (btype === 0) {
        // Uncompressed block
        bitBuf = 0; bitCount = 0;
        const len = input[inPos] | (input[inPos + 1] << 8);
        inPos += 4;
        for (let i = 0; i < len; i++) putByte(input[inPos++]);
      } else if (btype === 1 || btype === 2) {
        let litTree, distTree;
        if (btype === 1) {
          litTree = fixedLitTree;
          distTree = fixedDistTree;
        } else {
          const hlit = getBits(5) + 257;
          const hdist = getBits(5) + 1;
          const hclen = getBits(4) + 4;
          const codeLengths = new Uint8Array(19);
          for (let i = 0; i < hclen; i++) codeLengths[CODE_ORDER[i]] = getBits(3);
          const codeTree = buildHuffman(codeLengths, 7);

          const allLengths = new Uint8Array(hlit + hdist);
          let p = 0;
          while (p < hlit + hdist) {
            const sym = decodeSymbol(codeTree, 7);
            if (sym < 16) {
              allLengths[p++] = sym;
            } else if (sym === 16) {
              const rep = getBits(2) + 3;
              const prev = allLengths[p - 1];
              for (let i = 0; i < rep; i++) allLengths[p++] = prev;
            } else if (sym === 17) {
              const rep = getBits(3) + 3;
              for (let i = 0; i < rep; i++) allLengths[p++] = 0;
            } else if (sym === 18) {
              const rep = getBits(7) + 11;
              for (let i = 0; i < rep; i++) allLengths[p++] = 0;
            }
          }
          litTree = buildHuffman(allLengths.subarray(0, hlit), 15);
          distTree = buildHuffman(allLengths.subarray(hlit), 15);
        }

        while (true) {
          const sym = decodeSymbol(litTree, 15);
          if (sym === 256 || sym < 0) break;
          if (sym < 256) {
            putByte(sym);
          } else {
            const lengthIndex = sym - 257;
            const length = LENS[lengthIndex] + (LEXT[lengthIndex] > 0 ? getBits(LEXT[lengthIndex]) : 0);
            const distSym = decodeSymbol(distTree, 15);
            const dist = DISTS[distSym] + (DEXT[distSym] > 0 ? getBits(DEXT[distSym]) : 0);

            // Copy from sliding window
            for (let i = 0; i < length; i++) {
              // Locate byte from circular history
              let back = outPos - dist;
              let chunkIdx = outChunks.length;
              if (back < 0) {
                while (back < 0 && chunkIdx > 0) {
                  chunkIdx--;
                  back += outChunks[chunkIdx].length;
                }
                putByte(outChunks[chunkIdx][back]);
              } else {
                putByte(outChunk[back]);
              }
            }
          }
        }
      } else {
        throw new Error('Invalid deflate block type');
      }
    }

    // Assemble all output chunks
    const totalLen = outChunks.reduce((acc, c) => acc + c.length, 0) + outPos;
    const finalOut = new Uint8Array(totalLen);
    let offset = 0;
    for (const c of outChunks) {
      finalOut.set(c, offset);
      offset += c.length;
    }
    finalOut.set(outChunk.subarray(0, outPos), offset);
    return finalOut;
  }

  /**
   * Decompresses standard zlib (or raw deflate) payload in browser.
   */
  async function decompressZlib(compressedBytes) {
    if (typeof DecompressionStream !== 'undefined') {
      try {
        const format = (compressedBytes[0] === 0x78) ? 'deflate' : 'deflate-raw';
        const ds = new DecompressionStream(format);
        const writer = ds.writable.getWriter();
        writer.write(compressedBytes);
        writer.close();
        const resp = new Response(ds.readable);
        const buf = await resp.arrayBuffer();
        return new Uint8Array(buf);
      } catch (err) {
        // Fallback to JS decompressor
      }
    }
    return puffInflate(compressedBytes);
  }

  exports.decompressZlib = decompressZlib;
  exports.puffInflate = puffInflate;

})(typeof module !== 'undefined' && module.exports ? module.exports : (window.OTDInflate = {}));
