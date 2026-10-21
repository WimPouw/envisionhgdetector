// Analysis loading and validation, independent of the DOM and Plotly.
(() => {
  "use strict";
  const app = window.EnvisionDashboard ||= {};
function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  text = text.replace(/^\uFEFF/, ""); // remove byte-order-marker (BOM) if present
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') {
        field += '"';
        i++;
      } 
      else quoted = !quoted;
    } else if (c === "," && !quoted) {
      row.push(field);
      field = "";
    } 
    else if ((c === "\r" || c === "\n") && !quoted) {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      field = "";
    } 
    else field += c;
  }
  if (quoted) throw new Error("Unclosed quoted CSV field.");
  row.push(field);
  if (row.some((value) => value !== "")) rows.push(row);
  return rows;
}

function readRecords(text, required) {
  const [headers, ...rows] = parseCsv(text); // 1st row is headers
  if (
    headers &&
    (headers.some((key) => !key.trim()) ||
      new Set(headers).size !== headers.length)
  ) {
    throw new Error("CSV headers must be nonempty and unique.");
  }
  if (!headers || required.some((key) => !headers.includes(key))) {
    throw new Error(`Required columns: ${required.join(", ")}.`);
  }
  return rows.map((row) => {
    if (row.length !== headers.length)
      throw new Error("Unexpected CSV field count.");
    return Object.fromEntries(headers.map((key, i) => [key, row[i]]));
  });
}

function selectableFeatures(gestures) {
  return Object.keys(gestures[0] || {}).filter((key) => {
    if (key === "video_id") return false;
    if (key === "active_hand") return gestures.some(row => /^[LR]$/i.test(row[key].trim())); // make sure at least one valid L/R value exists
    // Keep partially valid features; plotting already excludes invalid rows.
    return gestures.some(row => {
      const raw = row[key].trim();
      return raw !== "" && Number.isFinite(Number(raw));
    });
  });
}

function featurePoints(gestures, feature) {
  return gestures.flatMap((gesture, index) => {
    const raw = gesture[feature].trim();
    if (feature === "active_hand") {
      const value = raw.toUpperCase();
      return /^[LR]$/.test(value) ? [{ index, value }] : [];
    }
    return raw && Number.isFinite(Number(raw))
      ? [{ index, value: Number(raw) }]
      : [];
  });
}

function similarityPoints(gestures, coordinates) {
  const byId = new Map();
  for (const row of coordinates) {
    if (
      !row.video_id.trim() ||
      byId.has(row.video_id) ||
      !row.x.trim() ||
      !row.y.trim() ||
      !Number.isFinite(Number(row.x)) ||
      !Number.isFinite(Number(row.y))
    ) {
      throw new Error(
        "Similarity coordinates must contain unique video IDs and finite x/y values.",
      );
    }
    byId.set(row.video_id, row);
  }
  return gestures.map((gesture, index) => {
    const row = byId.get(gesture.video_id);
    if (!row)
      throw new Error(
        `Missing similarity coordinates for ${gesture.video_id}.`,
      );
    return { index, x: Number(row.x), y: Number(row.y) };
  });
}

async function loadAnalysis(allFiles) {
  const files = allFiles.filter(file => {
    const parts = file.webkitRelativePath.split("/");
    return parts.length === 3 && parts[1] === "analysis";
  });
  if (!files.length) throw new Error("Choose the parent data folder containing analysis/ and videos/.");
  function getFile(name, required = true) {
    const matches = files.filter(file => file.name === name);
    if (matches.length > 1) throw new Error(`Multiple ${name} files found. Choose a single analysis folder.`);
    if (!matches.length && required) throw new Error(`Missing ${name}.`);
    return matches[0];
  }
  const featuresFile = getFile("kinematic_features.csv");
  const distancesFile = getFile("dtw_distances.npy");
  const coordinatesFile = getFile("gesture_visualization.csv", false);
  const gestures = readRecords(await featuresFile.text(), ["video_id"]);
  if (!gestures.length) throw new Error("No gestures found.");
  if (gestures.some(row => !row.video_id.trim()) || new Set(gestures.map(row => row.video_id)).size !== gestures.length) {
    throw new Error("Video IDs must be nonempty and unique.");
  }
  const distances = app.npy.readFloatMatrix(await distancesFile.arrayBuffer());
  const count = gestures.length;
  if (distances.rows !== count || distances.columns !== count) {
    throw new Error("DTW distances must be a square matrix matching the gesture count.");
  }
  for (const value of distances.data) {
    if (!Number.isNaN(value) && (!Number.isFinite(value) || value < 0)) {
      throw new Error("DTW distances must be nonnegative finite numbers or missing (NaN).");
    }
  }
  for (let i = 0; i < count; i++) {
    const diagonal = distances.get(i, i);
    if (!Number.isFinite(diagonal) || Math.abs(diagonal) > 1e-7) {
      throw new Error("DTW distances must have a zero diagonal.");
    }
    for (let j = i + 1; j < count; j++) {
      const forward = distances.get(i, j), reverse = distances.get(j, i);
      if (Number.isNaN(forward) || Number.isNaN(reverse)) {
        distances.set(i, j, NaN);
        distances.set(j, i, NaN);
      } else if (Math.abs(forward - reverse) > 1e-7 * Math.max(1, forward, reverse)) {
        throw new Error("DTW distances must be symmetric.");
      }
    }
  }
  const coordinates = coordinatesFile ? readRecords(await coordinatesFile.text(), ["video_id", "x", "y"]) : [];
  if (coordinatesFile && !coordinates.length) throw new Error("Similarity coordinates file is empty.");
  const projection = coordinates.length ? similarityPoints(gestures, coordinates) : [];
  const videos = allFiles.filter(file => {
    const parts = file.webkitRelativePath.split("/");
    return parts.length >= 3 && parts[1] === "videos" && /\.mp4$/i.test(file.name);
  });
  return { gestures, distances, projection, videos };
}
app.data = { loadAnalysis, selectableFeatures, featurePoints };
})();
