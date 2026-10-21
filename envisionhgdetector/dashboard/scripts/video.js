// Video variant selection and object-URL lifecycle.
(() => {
  "use strict";
  const app = window.EnvisionDashboard ||= {};
function findVideos(files, videoId) {
  if (!videoId) return [];
  return files.filter(file => {
    if (!file.name.startsWith(videoId) || !/\.mp4$/i.test(file.name)) return false;
    // A suffix must have a delimiter: id_1 must not match id_10.
    const suffix = file.name.slice(videoId.length);
    return /^\.mp4$/i.test(suffix) || /^[_-]/.test(suffix);
  });
}

function chooseVideo(matches, videoId, variant) {
  const candidates = matches.filter(
    (file) => /_tracked\.mp4$/i.test(file.name) === (variant === "tracked"),
  );
  const expectedStem = `${videoId}${variant === "tracked" ? "_tracked" : ""}`;
  const exact = candidates.filter(file => file.name.slice(0, -4) === expectedStem);
  const preferred = exact.length ? exact : candidates;
  return { file: preferred.length === 1 ? preferred[0] : null, ambiguous: preferred.length > 1 };
}

function createVideoPlayer({ player, title, status, summary, onError }) {
  let variant = "tracked", files = [], gesture = null, playingFile = null, url = null, requestVersion = 0;
  let skippedFolders = [];
  function updateSummary() {
    if (!summary) return;
    const messages = [];
    if (skippedFolders.length) messages.push(`${skippedFolders.length} video folder(s) could not be read.`);
    summary.textContent = messages.join(" ");
    summary.hidden = messages.length === 0;
    summary.title = skippedFolders.join("\n");
  }
  function clear() {
    requestVersion++;
    player.pause(); player.removeAttribute("src"); player.load(); player.hidden = true;
    if (url) URL.revokeObjectURL(url);
    url = null; playingFile = null;
  }
  async function render() {
    title.textContent = gesture ? gesture.video_id : "No gesture selected";
    const { file, ambiguous } = chooseVideo(findVideos(files, gesture?.video_id), gesture?.video_id, variant);
    const label = variant === "tracked" ? "Tracked" : "Untracked";
    status.textContent = !gesture ? "Select a gesture in a plot to preview its video."
      : !files.length ? "No MP4 videos were found under videos/ in the selected data folder."
      : ambiguous ? `Multiple ${label.toLowerCase()} videos match ${gesture.video_id}. Keep one matching clip or use a unique exact filename.`
      : file ? "" : `${label} video unavailable for ${gesture.video_id}. Choose the other version to check for an available clip.`;
    if (file && file === playingFile) return;
    clear();
    if (!file) return;
    const ticket = requestVersion;
    status.textContent = "Loading video?";
    try {
      // Directory-picker entries are lazy references; file-input entries are Files.
      const playable = typeof file.getFile === "function" ? await file.getFile() : file;
      if (ticket !== requestVersion) return;
      playingFile = file;
      url = URL.createObjectURL(playable);
      player.src = url;
      player.hidden = false;
      status.textContent = "";
      player.play()?.catch(() => { /* Native controls remain available if autoplay is blocked. */ });
    } catch (_) {
      if (ticket !== requestVersion) return;
      clear();
      status.textContent = `${label} video unavailable for ${gesture.video_id}. Choose the other version to check for an available clip.`;
    }
  }
  for (const input of document.querySelectorAll('input[name="video-variant"]')) {
    input.addEventListener("change", () => { if (input.checked) { variant = input.value; render(); } });
  }
  player.addEventListener("error", () => {
    if (playingFile && player.error) onError("Cannot play this video. Try another matching clip or a browser-compatible encoding.");
  });
  window.addEventListener("pagehide", clear);
  window.addEventListener("pageshow", event => { if (event.persisted) render(); });
  return { update(nextFiles, nextGesture, nextSkippedFolders = []) {
    files = nextFiles;
    gesture = nextGesture;
    skippedFolders = nextSkippedFolders;
    updateSummary();
    render();
  } };
}
app.video = { createVideoPlayer };
})();
