// preload.js — secure IPC bridge. Keeps renderer sandboxed while exposing only needed APIs.
// When contextIsolation:false fallback exists, renderer can still use require('electron') directly,
// but this file enables a gradual migration to contextIsolation:true + sandbox.
const { contextBridge, ipcRenderer } = require("electron");

const api = {
  send: (ch, ...args) => {
    const allowed = ["resize-window","move-window","walk-window","hide-window","show-window","drag-move","dock"];
    if (allowed.includes(ch)) ipcRenderer.send(ch, ...args);
  },
  invoke: (ch, ...args) => {
    const allowedInvoke = ["get-screen-bounds"];
    if (allowedInvoke.includes(ch)) return ipcRenderer.invoke(ch, ...args);
    return Promise.reject(new Error("invoke blocked: "+ch));
  },
  on: (ch, fn) => {
    const allowed = ["behave","avatar","dock-edge"];
    if (allowed.includes(ch)) ipcRenderer.on(ch, (_e, ...a) => fn(...a));
  },
  off: (ch, fn) => {
    try { ipcRenderer.removeListener(ch, fn); } catch(e){}
  }
};

// contextBridge only when isolation enabled; otherwise fallback to window.api
try {
  contextBridge.exposeInMainWorld("api", api);
} catch(e) {
  // fallback for nodeIntegration:true mode
  global.api = api;
  if (typeof window !== "undefined") window.api = api;
}
module.exports = api;
