// Plotly rendering and plot events. Data loading and video playback live elsewhere.
(() => {
  "use strict";
  const app = (window.EnvisionDashboard ||= {});
  const { featurePoints } = app.data;
  const escape = (value) =>
    String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");

  // Plotly hover labels use SVG text, so wrap with explicit line breaks.
  const hoverMeasure = document.createElement("canvas").getContext("2d");
  function wrapFeatureHover(value) {
    const styles = getComputedStyle(document.documentElement);
    const maxWidth =
      parseFloat(styles.getPropertyValue("--feature-hover-max-width")) || 240;
    hoverMeasure.font = `12px ${getComputedStyle(document.body).fontFamily}`;
    const lines = [];
    let line = "";
    for (const character of String(value)) {
      if (line && hoverMeasure.measureText(line + character).width > maxWidth) {
        lines.push(escape(line));
        line = "";
      }
      line += character;
    }
    if (line) lines.push(escape(line));
    return lines.join("<br>");
  }

  function theme() {
    const styles = getComputedStyle(document.documentElement);
    const color = (name) => styles.getPropertyValue(`--color-${name}`).trim();
    return Object.fromEntries(
      [
        "text",
        "panel",
        "border",
        "accent",
        "axis",
        "axis-text",
        "grid",
        "statistics",
        "box",
        "mean",
        "violin",
        "gesture",
        "transparent",
      ].map((name) => [name, color(name)]),
    );
  }
  function layout(colors, revision) {
    return {
      autosize: true,
      paper_bgcolor: colors.transparent,
      plot_bgcolor: colors.transparent,
      font: {
        family: getComputedStyle(document.body).fontFamily,
        color: colors.text,
        size: 12,
      },
      showlegend: false,
      hovermode: "closest",
      hoverdistance: 18,
      hoverlabel: {
        bgcolor: colors.panel,
        bordercolor: colors.accent,
        font: { color: colors.text },
      },
      modebar: {
        bgcolor: colors.transparent,
        color: colors.border,
        activecolor: colors.accent,
      },
      uirevision: `dataset-${revision}`, // tells plotly to preserve zoom/pan state across updates
    };
  }
  function marker(points, state, colors) {
    return {
      color: points.map((point) =>
        point.index === state.selectedGestureIndex
          ? colors.accent
          : colors.gesture,
      ),
      size: points.map((point) =>
        point.index === state.selectedGestureIndex ? 13 : 9,
      ),
      line: {
        width: 1,
        color: points.map((point) =>
          point.index === state.selectedGestureIndex
            ? colors.accent
            : colors.border,
        ),
      },
    };
  }

  function createPlots({ similarity, features, onSelect }) {
    const featureCharts = new Map(),
      queues = new WeakMap(),
      bound = new WeakSet(),
      selectionByGraph = new WeakMap();
    const resizeTargets = new Set();
    let resizeFrame = null;
    const observer = new ResizeObserver((entries) => {
      entries.forEach((entry) => resizeTargets.add(entry.target));
      if (resizeFrame) return;
      resizeFrame = requestAnimationFrame(() => {
        resizeFrame = null;
        for (const graph of resizeTargets) {
          if (graph.isConnected && graph.clientWidth && graph._fullLayout)
            Plotly.Plots.resize(graph);
        }
        resizeTargets.clear();
      });
    });
    observer.observe(similarity);

    function bindEvents(graph) {
      if (bound.has(graph)) return;
      bound.add(graph);
      graph.on("plotly_click", (event) => {
        const point = event.points?.find(
          (point) => point.data.meta?.kind === "gesture",
        );
        const index = point?.customdata?.[0];
        if (Number.isInteger(index)) onSelect(index);
      });
      graph.on("plotly_afterplot", () =>
        keyboardPoints(graph, selectionByGraph.get(graph)),
      );
    }
    function keyboardPoints(graph, selectedIndex) {
      for (const node of graph.querySelectorAll("[data-gesture-index]")) {
        for (const attribute of [
          "data-gesture-index",
          "tabindex",
          "role",
          "aria-label",
          "aria-pressed",
        ])
          node.removeAttribute(attribute);
        node.onkeydown = null;
      }
      for (const trace of graph._fullData || []) {
        if (trace.meta?.kind !== "gesture") continue;
        const nodes = graph.querySelectorAll(
          `.trace${CSS.escape(trace.uid)} .points path.point`,
        );
        nodes.forEach((node, position) => {
          // Plotly may retain DOM order while reordering trace data on react().
          const datum = trace.customdata[node.__data__?.i ?? position];
          if (!datum) return;
          const index = datum[0];
          node.setAttribute("tabindex", "0");
          node.setAttribute("role", "button");
          node.setAttribute("aria-label", datum[1]);
          node.setAttribute("aria-pressed", String(index === selectedIndex));
          node.dataset.gestureIndex = String(index);
          node.onkeydown = async (event) => {
            if (event.key !== "Enter" && event.key !== " ") return;
            event.preventDefault();
            await onSelect(index);
            graph.querySelector(`[data-gesture-index="${index}"]`)?.focus();
          };
        });
      }
    }
    function draw(graph, traces, nextLayout, state, filename, compact = false) {
      selectionByGraph.set(graph, state.selectedGestureIndex);
      const previous = queues.get(graph) || Promise.resolve();
      const task = previous
        .catch(() => {})
        .then(async () => {
          if (!graph.isConnected) return;
          if (graph.data) Plotly.Fx.unhover(graph);
          await Plotly.react(graph, traces, nextLayout, {
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: compact
              ? ["select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d"]
              : [],
            toImageButtonOptions: { filename, format: "png", scale: 2 },
          });
          bindEvents(graph);
          keyboardPoints(graph, state.selectedGestureIndex);
        });
      queues.set(graph, task);
      return task;
    }
    function dispose(graph) {
      observer.unobserve(graph);
      const task = (queues.get(graph) || Promise.resolve())
        .catch(() => {})
        .then(() => {
          if (graph.data) Plotly.purge(graph);
          bound.delete(graph);
        });
      queues.set(graph, task);
    }
    function renderSimilarity(state) {
      similarity.hidden = state.projection.length === 0;
      if (!state.projection.length) {
        dispose(similarity);
        similarity.replaceChildren();
        return Promise.resolve();
      }
      observer.observe(similarity);
      const colors = theme(), traces = [];
      const selected = state.selectedGestureIndex;
      if (state.hasGestureSelection) {
        const source = state.projection.find(point => point.index === selected);
        const lineX = [], lineY = [], hoverX = [], hoverY = [], distances = [];
        for (const target of state.projection) {
          if (target.index === selected) continue;
          const distance = state.distances.get(selected, target.index);
          if (!Number.isFinite(distance)) continue;
          // Null separators keep all connections in one trace without joining them.
          lineX.push(source.x, target.x, null);
          lineY.push(source.y, target.y, null);
          const detail = [distance, escape(state.gestures[selected].video_id), escape(state.gestures[target.index].video_id)];
          // Interior targets retain line hover without covering gesture endpoints.
          for (let step = 1; step < 50; step++) {
            const t = step / 50;
            hoverX.push(source.x + (target.x - source.x) * t);
            hoverY.push(source.y + (target.y - source.y) * t);
            distances.push(detail);
          }
        }
        if (lineX.length) {
          traces.push({
            type: "scatter", mode: "lines", name: "Connections", uid: "distance-lines",
            x: lineX, y: lineY, connectgaps: false,
            line: { color: colors.border, width: 1 },
            meta: { kind: "distance-lines" }, hoverinfo: "skip",
          }, {
            type: "scatter", mode: "markers", name: "DTW distance", uid: "distance-hover",
            x: hoverX, y: hoverY, customdata: distances,
            marker: { size: 1, color: colors.transparent, opacity: 0 },
            meta: { kind: "distance" },
            hovertemplate: "%{customdata[1]} ? %{customdata[2]}<br>DTW distance: %{customdata[0]}<extra></extra>",
          });
        }
      }
      const points = state.projection
        .slice()
        .sort(
          (a, b) => Number(a.index === selected) - Number(b.index === selected),
        );
      traces.push({
        type: "scatter",
        mode: "markers",
        name: "Gestures",
        uid: "gestures",
        meta: { kind: "gesture" },
        ids: points.map((point) => String(point.index)),
        x: points.map((point) => point.x),
        y: points.map((point) => point.y),
        customdata: points.map((point) => [
          point.index,
          escape(state.gestures[point.index].video_id),
        ]),
        marker: marker(points, state, colors),
        hovertemplate: "%{customdata[1]}<br>X: %{x}, Y: %{y}<extra></extra>",
      });
      return draw(
        similarity,
        traces,
        {
          ...layout(colors, state.revision),
          margin: { l: 55, r: 20, t: 38, b: 50 },
          xaxis: {
            title: { text: "Dimension 1" },
            gridcolor: colors.grid,
            zeroline: false,
            color: colors["axis-text"],
            showline: true,
            linecolor: colors.axis,
          },
          yaxis: {
            title: { text: "Dimension 2" },
            gridcolor: colors.grid,
            zeroline: false,
            color: colors["axis-text"],
            showline: true,
            linecolor: colors.axis,
          },
        },
        state,
        "gesture-similarity",
      );
    }
    function renderFeatures(state, selectedFeatures) {
      for (const [feature, entry] of featureCharts) {
        if (!selectedFeatures.includes(feature)) {
          dispose(entry.graph);
          entry.card.remove();
          featureCharts.delete(feature);
        }
      }
      const colors = theme();
      return Promise.all(
        selectedFeatures.map((feature) => {
          let entry = featureCharts.get(feature);
          if (!entry) {
            const card = document.createElement("div");
            card.className = "plot";
            const graph = document.createElement("div");
            graph.className = "feature-chart";
            graph.setAttribute("role", "group");
            graph.setAttribute(
              "aria-label",
              `${feature.replaceAll("_", " ")} distribution`,
            );
            card.append(graph);
            features.append(card);
            entry = { card, graph };
            featureCharts.set(feature, entry);
            observer.observe(graph);
          }
          // Reappend in feature-button order without replacing the Plotly graph.
          features.append(entry.card);
          const points = featurePoints(state.gestures, feature).sort(
            (a, b) =>
              Number(a.index === state.selectedGestureIndex) -
              Number(b.index === state.selectedGestureIndex),
          );
          const categorical = feature === "active_hand";
          const previousView = categorical && entry.graph.layout?.uirevision === `dataset-${state.revision}`
            ? entry.graph.layout : null;
          const values = points.map((point) => point.value);
          if (!categorical) values.sort((a, b) => a - b);
          if (!values.length) return Promise.resolve();
          const label = feature.replaceAll("_", " "),
            min = values[0],
            max = values.at(-1);
          const traces = [];
          if (!categorical && min !== max)
            traces.push({
              type: "violin",
              x: values.map(() => 0),
              y: values,
              name: "Distribution",
              uid: "distribution",
              points: false,
              width: 0.8,
              spanmode: "hard",
              line: { color: colors.violin, width: 1 },
              fillcolor: colors.violin,
              opacity: 0.3,
              meta: { kind: "distribution" },
              // Leave hover/click targeting to the gesture markers over the density.
              hoverinfo: "skip",
            });
          if (!categorical) traces.push({
            type: "box",
            name: "Statistics",
            uid: "statistics",
            x: values.map(() => 0),
            y: values,
            width: 0.2,
            boxpoints: false,
            line: { color: colors.statistics, width: 1 },
            fillcolor: colors.box,
            opacity: 0.8,
            meta: { kind: "statistics" },
            hoverinfo: "skip",
          });
          traces.push({
            type: "scatter",
            mode: "markers",
            name: "Gestures",
            uid: "gestures",
            meta: { kind: "gesture" },
            ids: points.map((point) => String(point.index)),
            x: points.map(
              (point) => -0.15 + ((point.index * 0.61803398875) % 1) * 0.3,
            ), // jitter to avoid overlap. between -0.15 and 0.15. change 0.3 for spread factor
            y: points.map((point) => point.value),
            marker: marker(points, state, colors),
            customdata: points.map((point) => [
              point.index,
              escape(state.gestures[point.index].video_id),
              `${wrapFeatureHover(state.gestures[point.index].video_id)}<br>${wrapFeatureHover(`${label}: ${point.value}`)}`,
            ]),
            hovertemplate: "%{customdata[2]}<extra></extra>",
            hoverlabel: { font: { size: 12 } },
          });
          if (!categorical) {
            const mean =
              values.reduce((sum, value) => sum + value, 0) / values.length;
            // Sample the line for hover across its width; gesture markers stay on top.
            traces.splice(traces.length - 1, 0, {
              type: "scatter",
              mode: "lines",
              name: "Mean",
              uid: "mean",
              meta: { kind: "mean" },
              x: Array.from({ length: 25 }, (_, index) => -0.12 + index * 0.01),
              y: Array(25).fill(mean),
              line: { color: colors.mean, width: 1, dash: "dot" },
              hovertemplate: "Mean: %{y:.6g}<extra></extra>",
            });
          }
          return draw(
            entry.graph,
            traces,
            {
              ...layout(colors, state.revision),
              margin: { l: 48, r: 12, t: 38, b: 20 },
              title: { text: escape(label), font: { size: 14 }, x: 0.5 },
              xaxis: {
                range: previousView?.xaxis.range?.slice() || [-0.55, 0.55],
                ...(categorical ? { autorange: previousView?.xaxis.autorange ?? false } : {}),
                showticklabels: false,
                showgrid: false,
                zeroline: false,
                fixedrange: !categorical,
              },
              yaxis: {
                ...(categorical ? {
                  type: "category", categoryorder: "array", categoryarray: ["L", "R"],
                  tickmode: "array", tickvals: ["L", "R"], ticktext: ["L", "R"],
                  range: previousView?.yaxis.range?.slice() || [-0.5, 1.5],
                  autorange: previousView?.yaxis.autorange ?? false, fixedrange: false,
                } : {}),
                gridcolor: colors.grid,
                zeroline: false,
                tickfont: { size: 10, color: colors["axis-text"] },
              },
            },
            state,
            feature,
            true,
          );
        }),
      );
    }
    return { renderSimilarity, renderFeatures };
  }
  app.plots = { createPlots };
})();
