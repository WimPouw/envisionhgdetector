# Dashboard scripts

- `theme.js`: applies the saved theme before the page paints and persists theme changes.
- `main.js`: entry point, shared gesture selection, and feature controls.
- `npy.js`: float32/float64 NPY reader using typed arrays (no pickle or eval).
- `data.js`: CSV parsing, NPY matrix loading, and analysis validation.
- `folders.js`: folder selection, saved directory handles, and automatic restore.
- `plots.js`: Plotly similarity and kinematic charts, hover/click/keyboard events.
- `video.js`: tracked/source preference and video playback.
- `vendor/plotly-3.7.0.min.js`: pinned Plotly.js distribution, MIT licensed; copied from the installed Plotly Python package's `package_data/plotly.min.js`. Upstream: https://github.com/plotly/plotly.js/releases/tag/v3.7.0

Only directory **handles** are persisted in IndexedDB, never dataset contents.
Theme preference is saved separately in localStorage.
To disable automatic restoration, set `AUTO_RESTORE_FOLDER` in `main.js`.

For persistent folder access, use Chrome or Edge and run from the project root:

```powershell
python -m http.server 8000 --bind 127.0.0.1 --directory envisionhgdetector/dashboard
```
Where `8000` is the port number and `127.0.0.1` is the host. 
Open http://localhost:8000/dashboard.html.

### Note:
- Missing DTW comparisons (`NaN` in the NPY matrix) do not block dataset loading.
Their similarity connections and distance tooltips are omitted. If either side
of a pair is missing, that pair is unavailable in both directions. Negative,
infinite, asymmetric finite distances and invalid diagonals still
produce validation errors. The matrix must still match the gesture count.

- Directory-picker loading enumerates folders and their contents.The full DTW matrix has a memory cost.