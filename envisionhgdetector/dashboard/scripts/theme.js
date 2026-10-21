// Apply the saved theme before the page paints, including direct file opening.
(() => {
  "use strict";
  let theme;
  try { theme = localStorage.getItem("envision-dashboard-theme"); } catch (_) {}
  if (theme !== "light" && theme !== "dark") {
    theme = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  document.documentElement.dataset.theme = theme;
  window.EnvisionDashboard ||= {};
  window.EnvisionDashboard.setTheme = next => {
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("envision-dashboard-theme", next); } catch (_) {}
  };
})();
