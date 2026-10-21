// Persist only a directory handle; dataset contents are never copied to storage.
// File-input selection lasts for the current session. Theme uses localStorage.
(() => {
  "use strict";
  const app = (window.EnvisionDashboard ||= {});
  function savedDirectory(action, handle, key = "data-folder") {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open("envision-dashboard", 1);
      request.onupgradeneeded = () =>
        request.result.createObjectStore("folders");
      request.onerror = () => reject(request.error);
      request.onblocked = () =>
        reject(new Error("Folder storage is blocked by another tab."));
      request.onsuccess = () => {
        const db = request.result;
        const transaction = db.transaction(
          "folders",
          action === "get" ? "readonly" : "readwrite",
        );
        const store = transaction.objectStore("folders");
        let operation;
        try {
          operation =
            action === "get"
              ? store.get(key)
              : action === "delete"
                ? store.delete(key)
                : store.put(handle, key);
        } catch (error) {
          db.close();
          reject(error);
          return;
        }
        transaction.oncomplete = () => {
          db.close();
          resolve(operation.result);
        };
        transaction.onabort = transaction.onerror = () => {
          db.close();
          reject(transaction.error);
        };
      };
    });
  }

  async function directoryFiles(root) {
    // Match folder names regardless of case and report the actual missing child.
    const children = new Map();
    for await (const entry of root.values()) {
      if (entry.kind === "directory") children.set(entry.name.toLowerCase(), entry);
    }
    const analysis = children.get("analysis");
    const videos = children.get("videos");
    if (!analysis) {
      const found = [...children.values()].map(entry => entry.name).join(", ") || "none";
      throw new Error(`Selected folder "${root.name}" has no analysis/ subfolder. Select the parent containing analysis/ and videos/. Subfolders found: ${found}`);
    }
    const files = [], skippedVideoFolders = [];
    async function collect(directory, path, recursive) {
      for await (const entry of directory.values()) {
        if (entry.kind === "directory") {
          if (recursive) {
            // Video directories may disappear or be inaccessible; keep other clips.
            try { await collect(entry, `${path}/${entry.name}`, true); }
            catch (_) { skippedVideoFolders.push(`${path}/${entry.name}`); }
          }
        } else if (
          recursive ? /\.mp4$/i.test(entry.name) : /\.(csv|npy)$/i.test(entry.name)
        ) {
          if (recursive) {
            // Keep only a reference; obtain the actual File when played.
            files.push({ name: entry.name, webkitRelativePath: `${path}/${entry.name}`, getFile: () => entry.getFile() });
            continue;
          }
          let file;
          try { file = await entry.getFile(); }
          catch (error) {
            throw new Error(`Cannot read "${path}/${entry.name}" (${error.name}): ${error.message}`);
          }
          Object.defineProperty(file, "webkitRelativePath", {
            value: `${path}/${entry.name}`,
          });
          files.push(file);
        }
      }
    }
    await collect(analysis, `${root.name}/analysis`, false);
    if (videos) {
      try { await collect(videos, `${root.name}/videos`, true); }
      catch (_) { skippedVideoFolders.push(`${root.name}/videos`); }
    }
    return { files, skippedVideoFolders };
  }

  function createFolderLoader({
    input,
    button,
    reconnectButton,
    onLoad,
    onReset = async () => {},
    onError,
    autoRestore = true,
  }) {
    let version = 0, pendingHandle = null;
    function clearReconnect() {
      pendingHandle = null;
      if (reconnectButton) reconnectButton.hidden = true;
    }
    const supportsHandles =
      location.protocol !== "file:" &&
      typeof window.showDirectoryPicker === "function";
    if (!supportsHandles) button.title = "Choose data folder (session only in this browser; Chrome or Edge on localhost supports reconnect)";
    async function loadFiles(files, ticket = ++version, skippedVideoFolders = [], reset = true) {
      try {
        if (ticket !== version) return false;
        if (reset) await onReset();
        if (ticket !== version) return false;
        const analysis = await app.data.loadAnalysis(files);
        if (ticket !== version) return false;
        await onLoad({ ...analysis, skippedVideoFolders });
        return true;
      } catch (error) {
        if (ticket === version) {
          await onReset();
          if (ticket === version) onError(`Could not load analysis: ${error.message}`);
        }
        return false;
      }
    }
    async function loadDirectory(handle, ticket) {
      try {
        if (ticket !== version) return false;
        await onReset();
        if (ticket !== version) return false;
        const { files, skippedVideoFolders } = await directoryFiles(handle);
        if (ticket !== version) return false;
        return await loadFiles(files, ticket, skippedVideoFolders, false);
      } catch (error) {
        if (ticket === version)
          onError(
            `Could not read folder "${handle.name}": ${error.message}`,
          );
        return false;
      }
    }
    reconnectButton?.addEventListener("click", async () => {
      if (!pendingHandle) return;
      const handle = pendingHandle, ticket = ++version;
      reconnectButton.disabled = true;
      try {
        // Request directly from the click so browser user activation is retained.
        const permission = await handle.requestPermission({ mode: "read" });
        if (ticket !== version) return;
        if (permission !== "granted") {
          onError("Folder access was not granted. Reconnect again or choose a data folder.");
          return;
        }
        if (await loadDirectory(handle, ticket) && ticket === version) clearReconnect();
      } catch (error) {
        if (ticket === version) onError(`Could not reconnect saved folder: ${error.message}. You can choose a data folder instead.`);
      } finally {
        reconnectButton.disabled = false;
      }
    });
    button.addEventListener("click", async () => {
      const ticket = ++version;
      if (!supportsHandles) {
        input.click();
        return;
      }
      try {
        const handle = await window.showDirectoryPicker({
          id: "envision-data",
          mode: "read",
        });
        if (!(await loadDirectory(handle, ticket)) || ticket !== version) return;
        clearReconnect();
        try {
          await savedDirectory("put", handle);
          await savedDirectory("delete", undefined, "folder-snapshot");
        } catch {
          onError(
            "Folder loaded, but this browser could not save access for the next reload.",
          );
        }
      } catch (error) {
        if (error.name !== "AbortError") {
          onError(
            "Persistent folder access is unavailable here. Choose the folder using the file picker instead.",
          );
          input.hidden = false;
        }
      }
    });
    input.addEventListener("change", async (event) => {
      const files = Array.from(event.target.files);
      if (!files.length) return;
      const ticket = ++version;
      const loaded = await loadFiles(files, ticket);
      if (loaded && ticket === version) {
        clearReconnect();
        try {
          // A session-only selection supersedes any previously saved folder.
          await savedDirectory("delete");
        } catch {
          onError("Folder loaded for this session, but the previous saved folder could not be cleared.");
        }
      }
      event.target.value = "";
    });
    async function restore() {
      try {
        // Delete legacy file copies without reading their potentially large blobs.
        // Run even when automatic restoration is disabled.
        await savedDirectory("delete", undefined, "folder-snapshot");
        if (!autoRestore || !supportsHandles || version) return;
        const handle = await savedDirectory("get");
        if (version || !handle) return;
        const permission = await handle.queryPermission({ mode: "read" });
        if (version) return;
        if (permission === "granted") await loadDirectory(handle, ++version);
        else {
          pendingHandle = handle;
          if (reconnectButton) {
            reconnectButton.hidden = false;
            reconnectButton.title = `Reconnect ${handle.name || "saved folder"}`;
          } else onError("Your browser needs permission to reopen the saved folder. Choose data folder to grant access again.");
        }
      } catch (error) {
        if (!version)
          onError(
            `Could not restore saved data: ${error?.message || error?.name || "unknown error"}. You can still choose a data folder.`,
          );
      }
    }
    return { restore };
  }
  app.folders = { createFolderLoader };
})();
