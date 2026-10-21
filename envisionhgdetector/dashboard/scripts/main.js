// Entry point: shared selection state and document controls.
// Scripts are loaded in order by dashboard.html so direct file opening works.
(() => {
  "use strict";
  const AUTO_RESTORE_FOLDER = true;
  const { data, folders, plots, video } = window.EnvisionDashboard;
  const el = (id) => document.getElementById(id);
  const showError = (message) => window.alert(message);
  if (!window.Plotly) {
    showError(
      "Plotly could not load. Keep scripts/vendor/plotly-3.7.0.min.js beside the dashboard scripts.",
    );
    return;
  }
  const state = {
    gestures: [],
    distances: null,
    projection: [],
    videos: [],
    selectedGestureIndex: null,
    hasGestureSelection: false,
    revision: 0,
  };
  const player = video.createVideoPlayer({
    player: el("video-player"),
    title: el("selected-gesture-name"),
    status: el("video-status"),
    summary: el("video-load-summary"),
    onError: showError,
  });
  const charts = plots.createPlots({
    similarity: el("similarity-plot"),
    features: el("kinematic-plots"),
    onSelect: selectGesture,
  });

  function updateThemeButton() {
    const label =
      document.documentElement.dataset.theme === "dark"
        ? "Switch to light mode"
        : "Switch to dark mode";
    el("theme-toggle").setAttribute("aria-label", label);
    el("theme-toggle").title = label;
  }
  updateThemeButton();
  el("theme-toggle").addEventListener("click", () => {
    window.EnvisionDashboard.setTheme(
      document.documentElement.dataset.theme === "dark" ? "light" : "dark",
    );
    updateThemeButton();
    render().catch((error) =>
      showError(`Could not update plots: ${error.message}`),
    );
  });

  function selectedFeatures() {
    return Array.from(
      el("feature-options").querySelectorAll('.feature-checkbox:checked'),
      (input) => input.value,
    );
  }
  function renderFeatures() {
    const selected = selectedFeatures();
    const total =
      el("feature-options").querySelectorAll(".feature-checkbox").length;
    el("toggle-all-features").disabled = total === 0;
    el("toggle-all-features").textContent =
      total > 0 && selected.length === total ? "Deselect all" : "Select all";
    el("plot-status").textContent = state.gestures.length
      ? ""
      : "Load analysis to view features.";
    for (const group of el("feature-options").querySelectorAll(".feature-dropdown")) {
      const inputs = group.querySelectorAll(".feature-checkbox");
      const count = group.querySelectorAll(".feature-checkbox:checked").length;
      group.querySelector(".feature-count").textContent = `${count}/${inputs.length} selected`;
    }
    return charts.renderFeatures(state, selected);
  }
  async function render() {
    player.update(state.videos, state.gestures[state.selectedGestureIndex], state.skippedVideoFolders);
    el("similarity-status").textContent = state.gestures.length
      ? ""
      : "Load analysis to compare gestures.";
    await Promise.all([charts.renderSimilarity(state), renderFeatures()]);
  }
  async function selectGesture(index) {
    if (!Number.isInteger(index) || !state.gestures[index]) return;
    state.selectedGestureIndex = index;
    state.hasGestureSelection = true;
    try {
      await render();
    } catch (error) {
      showError(`Could not update plots: ${error.message}`);
    }
  }
  function setupFeatures() {
    const groups = new Map(
      [
        "Hand",
        "Elbow",
        "Timing",
        "Spatial",
        "Other",
      ].map((name) => [name, []]), // update the feature mapping here
    );
    for (const feature of data.selectableFeatures(state.gestures)) {
      const group = feature === "active_hand" || feature.startsWith("hand_")
        ? "Hand"
        : feature.startsWith("elbow_")
          ? "Elbow"
          : feature === "duration" || feature.startsWith("hold_")
            ? "Timing"
            : ["space_use", "volume", "max_height"].includes(feature) ||
                feature.startsWith("mcneillian_")
              ? "Spatial"
              : "Other";
      groups.get(group).push(feature);
    }
    const rows = [];
    for (const [name, groupFeatures] of groups) {
      if (!groupFeatures.length) continue;
      const row = document.createElement("details");
      row.className = "feature-dropdown";
      const heading = document.createElement("summary");
      const title = document.createElement("span");
      title.textContent = name;
      const count = document.createElement("span");
      count.className = "feature-count";
      heading.append(title, count);
      heading.addEventListener("click", () => {
        for (const other of el("feature-options").querySelectorAll("details[open]")) {
          if (other !== row) other.open = false;
        }
      });
      const options = document.createElement("div");
      options.className = "feature-menu";
      options.setAttribute("role", "group");
      options.setAttribute("aria-label", name);
      for (const feature of groupFeatures) {
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.className = "feature-checkbox";
        input.value = feature;
        input.checked = true;
        input.addEventListener("change", () => {
          renderFeatures().catch((error) =>
            showError(`Could not update plots: ${error.message}`),
          );
        });
        label.append(input, document.createTextNode(feature.replaceAll("_", " ")));
        options.append(label);
      }
      row.append(heading, options);
      rows.push(row);
    }
    el("feature-options").replaceChildren(...rows);
  }
  document.addEventListener("click", (event) => {
    for (const group of el("feature-options").querySelectorAll("details[open]")) {
      if (!group.contains(event.target)) group.open = false;
    }
  });
  el("feature-options").addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const group = event.target.closest("details[open]");
    if (group) {
      group.open = false;
      group.querySelector("summary").focus();
      event.preventDefault();
    }
  });
  el("toggle-all-features").addEventListener("click", () => {
    const buttons = Array.from(
      el("feature-options").querySelectorAll(".feature-checkbox"),
    );
    const selectAll = buttons.some(
      (input) => !input.checked,
    );
    buttons.forEach((input) => { input.checked = selectAll; });
    renderFeatures().catch((error) =>
      showError(`Could not update plots: ${error.message}`),
    );
  });
  const loader = folders.createFolderLoader({
    input: el("data-folder"),
    button: el("choose-data-folder"),
    reconnectButton: el("reconnect-data-folder"),
    autoRestore: AUTO_RESTORE_FOLDER,
    onError: showError,
    async onReset() {
      Object.assign(state, {
        gestures: [], distances: null, projection: [], videos: [], skippedVideoFolders: [],
        selectedGestureIndex: null, hasGestureSelection: false, revision: state.revision + 1,
      });
      el("feature-options").replaceChildren();
      await render();
    },
    async onLoad(analysis) {
      Object.assign(state, analysis, {
        selectedGestureIndex: null,
        hasGestureSelection: false,
        revision: state.revision + 1,
      });
      setupFeatures();
      await render();
    },
  });
  loader.restore();
})();
