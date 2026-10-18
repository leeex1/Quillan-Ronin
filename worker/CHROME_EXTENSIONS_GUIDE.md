# Chrome Extensions — Practical Guide for Quillan-Ronin (Manifest V3)

Source: https://developer.chrome.com/docs/extensions + subpages fetched 2026-08-26. See also `extension/manifest.json:1`, `extension/background.js:1`, `extension/content.js:1`.

---

## 1. Manifest V3 Essentials

Quillan-Ronin uses `manifest_version: 3` with `service_worker` background.

```json
{
  "manifest_version": 3,
  "name": "Quillan-Ronin Agent",
  "version": "1.4.0",
  "background": { "service_worker": "background.js" },
  "permissions": ["activeTab","scripting","sidePanel","debugger","tabs","storage","alarms","webNavigation"],
  "host_permissions": ["http://localhost:7777/*","http://127.0.0.1:7777/*","<all_urls>"],
  "content_scripts": [{ "matches": ["<all_urls>"], "js": ["content.js"], "run_at": "document_idle" }],
  "action": { "default_title": "Quillan-Ronin" },
  "side_panel": { "default_path": "popup.html" }
}
```

* **MV3 vs MV2:** background pages → event-based `service_worker`; no remotely-hosted code; `chrome.action` replaces `browserAction/pageAction`. See https://developer.chrome.com/docs/extensions/develop/migrate/what-is-mv3
* **Icons:** declare under `icons` and `action.default_icon`.

---

## 2. Architecture You Use

### Service Worker (`background.js`)
* Event-based, suspends after ~30s idle. Keep alive via `chrome.alarms` (0.4 min heartbeat) + `chrome.webNavigation.onCompleted` re-arm — exactly what Quillan does `background.js:30`.
* Has access to `chrome.tabs`, `chrome.scripting`, `chrome.storage`, `chrome.alarms`, `chrome.offscreen`, `chrome.sidePanel`, `chrome.debugger`. **No DOM**.
* Use `chrome.runtime.onInstalled` / `onStartup` to `ensureOffscreenDocument()` + `initializeTabContext()` + `startPolling()`.

### Content Script (`content.js`)
* Runs in **isolated world** (DOM shared, JS vars isolated). Can read/modify DOM, add `data-quillan-idx` badges.
* Declared statically via `manifest.content_scripts` (`<all_urls>`, `document_idle`) or dynamically via `chrome.scripting.registerContentScripts` or programmatically via `chrome.scripting.executeScript`.
* Limited APIs directly: `chrome.runtime.sendMessage/onMessage`, `chrome.storage`, `chrome.i18n`, `chrome.dom`. Everything else via messaging to service worker.
* **Match patterns:** `https://*.nytimes.com/*`, `<all_urls>`. Combine with `exclude_matches`, `include_globs`, `exclude_globs`, `all_frames`, `run_at` (`document_start|document_end|document_idle`).
* **Web-accessible resources:** expose via `web_accessible_resources` + `chrome.runtime.getURL()`.

### Popup / Side Panel (`popup.html/js`)
* `chrome.sidePanel.setPanelBehavior({openPanelOnActionClick:true})` must be called from **service worker** (not popup) — popup now avoids it (was crashing).
* Side panel `default_path: popup.html` persists while open, good for streaming chat.

### Offscreen Document (`offscreen.html/js`)
* Needed for DOMParser/workers. Create via `chrome.offscreen.createDocument({url, reasons:['DOM_PARSER','WORKERS'], justification})`. Check existence via `chrome.runtime.getContexts({contextTypes:['OFFSCREEN_DOCUMENT']})`.

---

## 3. Messaging (Service Worker ↔ Content ↔ Popup)

* **One-off:** `chrome.tabs.sendMessage(tabId, {action:'getPageContent'})` + `chrome.runtime.onMessage.addListener((req,sender,sendResponse)=>{...; return true})`
* **Long-lived:** `chrome.tabs.connect(tabId)` → `runtime.Port`.
* **To embedding page:** `window.postMessage` (shared DOM), content script bridges to extension.

Quillan uses `chrome.runtime.onMessage` for `getPageContent`, `indexInteractiveElements`, `clickElement`, `typeElement`, `scrollTo`, plus `startPolling/stopPolling` and `pageReady` signals.

---

## 4. Tabs & Scripting (Core for Browsing)

### `chrome.tabs` (service worker / popup only)
* `query({active:true, currentWindow:true})` → current tab (`tabs:Tab` needs `tabs` perm for `url/title/favIconUrl`, or `host_permissions`/`activeTab`).
* `create({url, active:true})` vs `update(tabId,{url})` — Quillan now uses **single-tab** `update` to avoid spam (user preference).
* `sendMessage(tabId, msg)`, `captureVisibleTab(windowId, {format:'png'})` (max 2/sec), `get(tabId)`, `onActivated/onUpdated/onRemoved`.
* **ActiveTab:** temporary host perm on user gesture (click action) — no warning.

### `chrome.scripting`
* `executeScript({target:{tabId}, func, args, files})` — fallback when `sendMessage` fails on `chrome://` pages. Injected func must be self-contained (no closure vars).
* `registerContentScripts/update/unregisterContentScripts` for dynamic injection.

---

## 5. Permissions You Request

| Permission | Why Quillan needs it |
|---|---|
| `activeTab` | Read active tab on click without broad host warning |
| `tabs` | Query `url/title/favIconUrl` for context tracking |
| `scripting` | Fallback `executeScript` when content script unreachable |
| `storage` + `unlimitedStorage` | Persist `tab_<id>` contexts + chat history |
| `alarms` | Keepalive `quillan-keepalive` 0.4 min |
| `webNavigation` | `onCompleted` to restart polling after suspend |
| `debugger` | Fallback `Runtime.evaluate` when scripting blocked |
| `offscreen` | DOMParser/workers |
| `sidePanel` | Side panel UI |
| `<all_urls>` host | Content script on any site + host perm for `captureVisibleTab`/`scripting` |

See full list: https://developer.chrome.com/docs/extensions/reference/permissions-list

---

## 6. Storage, Cookies, Network

* `chrome.storage.local` (Quillan: `tab_<id>` contexts) + `sync`/`session`/`managed`. Event `chrome.storage.onChanged`.
* `chrome.cookies`, `chrome.webRequest`/`declarativeNetRequest`/`webNavigation` for observing/blocking requests.
* Cross-origin `fetch` to `localhost:7777` requires `host_permissions` (already `http://localhost:7777/*`) and CORS headers on server (moved to top of handler `server.js:290`).

---

## 7. UI Options

* **Action** (`chrome.action`): toolbar icon, badge, popup.
* **Side Panel** (`chrome.sidePanel`): persistent alongside page — used by Quillan.
* **Context Menus** (`chrome.contextMenus`), **Omnibox** (`chrome.omnibox`), **Commands** (`chrome.commands`), **Notifications** (`chrome.notifications`).
* **Override pages:** `chrome_url_overrides` for newtab/history/bookmarks.

---

## 8. Recent Relevant What's New

* Chrome 153: `browser.publicSuffix` API, default pinning for extensions.
* Chrome 148: `browser` namespace (promises, `browser.*`) alongside `chrome`.
* Built-in AI Challenge / Prompt API for extensions — see https://developer.chrome.com/docs/extensions/ai/prompt-api

---

## 9. Debugging Checklist for Quillan

1. `brave://extensions` → Developer mode → Load unpacked → Inspect views: service worker → console for `[Quillan Extension] Started/poll`.
2. If `chrome://` page → `Content script not reachable` is expected (fallback to `scripting` fails too).
3. Keep sidePanel open to keep service worker alive; `alarms` heartbeat logs `Keepalive alarm — restarting poll`.
4. Network: `http://127.0.0.1:7777/api/ext/poll` must show `Access-Control-Allow-Origin: *` (fixed by moving CORS to top).
5. After manifest edits, bump `version` or click Reload.

---

## 10. Fully Web Use + Browser Computer Use — Pulled from linked docs (your request)

**Quillan-Ronin now requests:** `tabCapture, desktopCapture, cookies, history, bookmarks, downloads, browsingData, topSites, sessions, windows, tabGroups, contextMenus, commands, notifications, identity, privacy, pageCapture, declarativeNetRequest, webRequest, contentSettings, management, system.display` in `manifest.json:31` (+ existing `activeTab, scripting, sidePanel, debugger, tabs, storage, alarms, webNavigation`).

| Capability | API | How Quillan uses it |
|---|---|---|
| **Tab/window control** | `tabs, windows, tabGroups` | `create/update/query/sendMessage/captureVisibleTab` — single-tab `update` for YouTube/Google search, keep `lastSearchTabId` |
| **Screen media** | `tabCapture.getMediaStreamId({targetTabId})` | After user gesture → `MediaStream` via `getUserMedia({chromeMediaSource:'tab', chromeMediaSourceId:id})`; `onStatusChanged` |
| **Desktop picker** | `desktopCapture.chooseDesktopMedia(['screen','window','tab'], tab, cb)` → `streamId` | Picker UI, `cancelChooseDesktopMedia`, 1-sec expiry — now in `background.js:captureDesktopMedia` |
| **Page MHTML** | `pageCapture.saveAsMHTML({tabId})` | Save full page as MHTML blob `background.js:capturePageMHTML` |
| **Web nav observe** | `webNavigation.onCompleted/onBeforeNavigate/...` | Re-arm polling after SW suspend `background.js:30` |
| **Cookies** | `cookies.getAll({url})`, `get/set/remove`, `onChanged`, partitioned `partitionKey` | `background.js:getCookies/setCookie` via `browser_read` grounding |
| **History/bookmarks/downloads** | `history.search`, `bookmarks.getTree/create`, `downloads.search`, `topSites.get`, `sessions.getRecentlyClosed` | `background.js:getHistory/getBookmarks/...` — now queryable via `sendCmd('getHistory')` etc. |
| **Storage** | `storage.local/sync/session/managed` | `tab_<id>` contexts already |
| **Net filter** | `declarativeNetRequest` + `webRequest` | Observe/block/modify — ready for future adblock / request mod |
| **Content settings / privacy** | `contentSettings[contentType].get`, `privacy`, `proxy` | `background.js:handleContentSettings` |
| **System** | `system.display.getInfo`, `commands.onCommand`, `contextMenus.create`, `notifications.create` | Keyboard shortcuts + right-click → `sidePanel.open` |

**Computer-use bridge:** `chrome.debugger.attach({tabId}, "1.3") → sendCommand("Runtime.evaluate" / "Page.navigate" / "DOM.*")` fallback in `background.js:389` when `scripting` blocked; plus `tabCapture`/`desktopCapture` streams consumable in `offscreen` doc (Chrome 116 `consumerTabId` same-origin). `nativeMessaging` ready to add `com.quillan.host` for OS-level if needed.

Sources fetched: `tabCapture`, `desktopCapture`, `webNavigation`, `cookies` docs (4) + prior `sidePanel`, `action`, `debugger`, `service-workers`, `content-scripts`, `tabs`.

---

## 11. Hyperlinked Deep-Dive (fetched from your pasted sections)

### Design the user interface — https://developer.chrome.com/docs/extensions/develop#design-the-user-interface
* **Side panel** `chrome.sidePanel` (Chrome 114+ MV3): host UI alongside page. `manifest.side_panel.default_path` for global, or `chrome.sidePanel.setOptions({tabId, path, enabled})` for site-specific (e.g. only on `google.com`). `setPanelBehavior({openPanelOnActionClick:true})` must be in service worker with `action` declared. `sidePanel.open({windowIdtabId})` requires user gesture (action click, contextMenu, command). Events `onOpened/onClosed`. Quillan fix: moved call from `popup.js` to `background.js:13`.
  Details: https://developer.chrome.com/docs/extensions/reference/api/sidePanel — includes `setOptions`, `getOptions`, `open({windowId|tabId})`, `close`, `getLayout`, pin UI.

* **Action** `chrome.action` (MV3): toolbar icon, badge (≤4 chars), tooltip, popup. `manifest.action.{default_icon,default_title,default_popup}`. `action.setIcon({path|imageData, tabId})`, `setBadgeText({text, tabId})`, `setBadgeBackgroundColor`, `setTitle`, `setPopup`, `enable/disable(tabId)`, `onClicked` (not fired if popup set), `declarativeContent.ShowAction()` to enable per-site.
  Details: https://developer.chrome.com/docs/extensions/reference/api/action

* **Menus** `chrome.contextMenus`: `create({id,title,contexts:['all','page','selection']})`, `onClicked` → `chrome.sidePanel.open({windowId: tab.windowId})` pattern (sample: cookbook.sidepanel-open).

### Control the browser — https://developer.chrome.com/docs/extensions/develop#control-the-browser
* **Override Chrome pages / Settings overrides:** `chrome_settings_overrides` + `chrome_url_overrides` (newtab, history, bookmarks manager) — HTML override.
* **Extending DevTools** + **`chrome.debugger`**: attach to tab `debugger.attach({tabId},"1.3")` → `sendCommand({tabId}, "Runtime.evaluate"|"Page.navigate"|"DOM.*", params)` → `detach`. Restricted to 27 domains (DOM, Network, Runtime, etc.). Needs `debugger` permission; shows infobar. Supports flat sessions `sessionId` for out-of-process iframes via `Target.setAutoAttach({flatten:true})`. Quillan uses as fallback when `scripting` blocked `background.js:391`.
  Details: https://developer.chrome.com/docs/extensions/reference/api/debugger
* **Notifications** `chrome.notifications`: templates + `notifications` permission.
* **History** `chrome.history` + `chrome.browsingData` + `chrome.topSites`.
* **Tabs / Windows** `chrome.tabs`/`tabGroups`/`windows`: see §4 + https://developer.chrome.com/docs/extensions/reference/api/tabs — `query`, `create/update`, `captureVisibleTab` (2/sec, needs `<all_urls>` or `activeTab`), `detectLanguage`, `group/ungroup`, `move` with drag guard.
* **Commands** `chrome.commands`: manifest `commands:{ "open-panel":{suggested_key,...}}}` → `commands.onCommand`.
* **Identity** `chrome.identity.getAuthToken` (OAuth2).
* **Management** `chrome.management` (list/enable/disable extensions).
* **Omnibox** `chrome.omnibox` (keyword → `onInputChanged/onInputEntered`).
* **Privacy / Proxy** `chrome.privacy` / `chrome.proxy`.
* **Downloads** `chrome.downloads`, **Bookmarks** `chrome.bookmarks`, **Reading List** `chrome.readingList`.

### Control the web — https://developer.chrome.com/docs/extensions/develop#control-the-web
* **Inject JS/CSS (Content scripts)** — isolated world, see full doc already in §2/§3. `run_at: document_idle` (default, between `document_end` and `onload`), `document_start`, `document_end`. `all_frames`, `match_origin_as_fallback` for `about:blank/data:/blob:` frames.
* **ActiveTab** `activeTab` perm: temporary host perm on user gesture, revoked on navigate/close — no warning.
* **Web requests** `declarativeNetRequest` (MV3 preferred) / `webRequest` / `webNavigation` (`onUpdated`, `onCompleted` — Quillan uses to re-arm polling).
* **Screen capture** `chrome.tabCapture` / `getDisplayMedia()` vs `tabs.captureVisibleTab`.
* **Content settings** `chrome.contentSettings`.

### Core concepts — https://developer.chrome.com/docs/extensions/develop#core-concepts
* **Service workers** `service-workers`: central event handler, loaded on demand, unloaded on idle (no DOM). Use `offscreen` for DOM. See deep doc: https://developer.chrome.com/docs/extensions/develop/concepts/service-workers — includes lifecycle (install/activate/idle shutdown), events, update flow.
  Details fetched: https://developer.chrome.com/docs/extensions/develop/concepts/service-workers — event page vs SW, cannot access DOM, use `runtime.onInstalled/onStartup`, `alarms` keepalive.
* **Permissions** `declare-permissions`: `permissions` vs `host_permissions` vs `optional_permissions`; `activeTab` no warning; `declarativeNetRequestFeedback` etc.
* **Content filtering:** `declarativeNetRequest` vs `webRequest` blocking.
* **Messaging:** `runtime.sendMessage`, `tabs.sendMessage`, `runtime.connect` (Port), `chrome.storage.onChanged`.
* **Native messaging** `nativeMessaging` (host `com.*`).
* **Avoid remotely hosted code:** MV3 must bundle all code.
* **Storage** `storage` API: `local/sync/session/managed` (4 areas) — Quillan uses `local` for `tab_<id>` + `unlimitedStorage`.
* **Offscreen** `chrome.offscreen` — see §2.
* **Cross-origin isolation:** `cross_origin_embedder_policy` / `cross_origin_opener_policy`.
* **Update lifecycle:** Chrome auto-updates, `runtime.onUpdateAvailable`.

---

## 11. Docs Fixes for Your Exact Issues (from links you pasted)

Based on `service-workers/lifecycle` + `declare-permissions` + `messaging` docs you highlighted:

| Your Issue | Doc Says | Fix Applied to Quillan |
|---|---|---|
| **Tools work 1-2 times then stop** | `service-workers/lifecycle:1` — SW terminates after **30s inactivity**, `fetch()` >30s, or single request >5min. **Must not keep alive indefinitely**, use `storage` not globals. Chrome 120 alarms now 30s min. | `background.js:30` `alarms.create 0.4min` heartbeat + `webNavigation.onCompleted` re-arm + `pollDelay` 400ms (not 4000) + `storage.local` for `tab_<id>` not globals. `manifest.json:41` adds `alarms,webNavigation,storage` |
| **Not listening / truncating breaks** | `messaging:1` — `runtime.onMessage` must `return true` (or promise) to keep channel open for async `sendResponse`; otherwise `undefined→null`. `declare-permissions` — `host_permissions` needed for `tabs.url/title` + `fetch` + `cookies`; `activeTab` only temp on gesture | `background.js:62` `onMessage` now `return true` for `getTabContext/getAllTabContexts/pageReady` + `content.js:8` `return true`; `manifest.json:31` has `host_permissions <all_urls>` + `activeTab` fallback via `scripting.executeScript` |
| **New tab spam vs single-tab** | `tabs` doc — `tabs.create` vs `tabs.update`; `declare-permissions` — `tabs` perm needed for `url/title` filtering | Quillan now `tabs.update(tabId,{url})` single-tab `popup.js:122` + `server.js:232` `browser_search` defaults Google (general) not YouTube, only `engine=youtube` if `video|youtube` in query |
| **Search bar vs URL bar vs AI prompt** | `content-scripts:1` — isolated world, DOM shared, use `executeScript({func,args})` self-contained; `sidePanel` needs `openPanelOnActionClick` in SW not popup | `popup.js:122` `handleUseCurrentSearch` uses `executeScript` typing into `input[name="q"], ytd-searchbox input, div[contenteditable]` + `Enter`, not `tabs.update` URL hack; `handlePromptModel` `popup.js:266` 2-pass (click AI chip → retry) staying in **open tab** per guide §2, not `tabs.update` to Google |
| **Hallucinated discography / VIDEO_ID** | `messaging:1` Serialization is `JSON.stringify` only — hallucinated `VIDEO_ID` fails `isYoutube` check; `declare-permissions` host perms needed for `cookies` etc. | `server.js:44` strict schema + `background.js:576` `collectYouTubeLinks` real `watch?v=` hrefs + `server.js:212` rejects `VIDEO_ID` placeholder |

*Lifecycle keepalive per doc:* Chrome 120 alarms 30s, Chrome 116 websocket extends SW, Chrome 114 `runtime.onConnect` keeps alive — Quillan uses alarms (Chrome 120) + `desktopCapture`/`identity` are allowed to exceed 5min (doc), but we keep polling <30s.

## 12. Links (expanded)

* Extensions home: https://developer.chrome.com/docs/extensions
* Develop overview: https://developer.chrome.com/docs/extensions/develop
* Content scripts: https://developer.chrome.com/docs/extensions/develop/concepts/content-scripts
* Tabs API: https://developer.chrome.com/docs/extensions/reference/api/tabs
* Side Panel: https://developer.chrome.com/docs/extensions/reference/api/sidePanel
* Action: https://developer.chrome.com/docs/extensions/reference/api/action
* Debugger: https://developer.chrome.com/docs/extensions/reference/api/debugger
* Service workers: https://developer.chrome.com/docs/extensions/develop/concepts/service-workers
* Messaging: https://developer.chrome.com/docs/extensions/develop/concepts/messaging
* Manifest reference: https://developer.chrome.com/docs/extensions/reference/manifest
* Permissions list: https://developer.chrome.com/docs/extensions/reference/permissions-list
* Offscreen: https://developer.chrome.com/docs/extensions/reference/api/offscreen
* Scripting: https://developer.chrome.com/docs/extensions/reference/api/scripting
* Storage: https://developer.chrome.com/docs/extensions/develop/concepts/storage-and-cookies
* Samples: https://developer.chrome.com/docs/extensions/samples
