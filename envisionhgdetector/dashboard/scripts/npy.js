// Numeric NPY reader, following numpy.lib.format; no Python evaluation or pickle.
(() => {
  "use strict";
  const app = window.EnvisionDashboard ||= {};
  const nativeLittleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

  function readFloatMatrix(buffer) {
    const bytes = new Uint8Array(buffer), view = new DataView(buffer);
    if (bytes.length < 10 || bytes[0] !== 0x93 || String.fromCharCode(...bytes.subarray(1, 6)) !== "NUMPY") {
      throw new Error("Invalid NPY file signature.");
    }
    const major = bytes[6];
    if (![1, 2, 3].includes(major) || bytes[7] !== 0) throw new Error("Unsupported NPY format version.");
    const start = major === 1 ? 10 : 12;
    if (bytes.length < start) throw new Error("Truncated NPY header.");
    const offset = start + (major === 1 ? view.getUint16(8, true) : view.getUint32(8, true));
    if (offset > bytes.length) throw new Error("Truncated NPY header.");
    const header = new TextDecoder(major === 3 ? "utf-8" : "latin1").decode(bytes.subarray(start, offset));
    const dtype = /['"]descr['"]\s*:\s*['"]([<>=|])f([48])['"]/.exec(header);
    const order = /['"]fortran_order['"]\s*:\s*(True|False)\b/.exec(header);
    const shape = /['"]shape['"]\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,?\s*\)/.exec(header);
    if (!dtype) throw new Error("DTW NPY must contain float32 or float64 numbers; object/pickled arrays are unsupported.");
    if (!order || !shape) throw new Error("DTW NPY must have a valid two-dimensional shape and storage order.");
    const rows = Number(shape[1]), columns = Number(shape[2]), itemSize = Number(dtype[2]);
    const count = rows * columns;
    if (!Number.isSafeInteger(count) || !Number.isSafeInteger(offset + count * itemSize) || offset + count * itemSize !== bytes.length) {
      throw new Error("NPY data size does not match its shape (truncated or extra data).");
    }
    if (dtype[1] === "|") throw new Error("DTW floating-point NPY must specify byte order.");
    const littleEndian = dtype[1] === "=" ? nativeLittleEndian : dtype[1] === "<";
    const ArrayType = itemSize === 4 ? Float32Array : Float64Array;
    let data;
    if (littleEndian === nativeLittleEndian && offset % itemSize === 0) {
      // Normal NumPy files use a direct view: no second copy of matrix values.
      data = new ArrayType(buffer, offset, count);
    } else {
      data = new ArrayType(count);
      for (let i = 0; i < count; i++) {
        data[i] = itemSize === 4 ? view.getFloat32(offset + i * itemSize, littleEndian) : view.getFloat64(offset + i * itemSize, littleEndian);
      }
    }
    const index = order[1] === "True" ? (row, column) => column * rows + row : (row, column) => row * columns + column;
    return {
      rows, columns, data,
      get: (row, column) => data[index(row, column)],
      set: (row, column, value) => { data[index(row, column)] = value; },
    };
  }
  app.npy = { readFloatMatrix };
})();
