let bubble, msgs, input, sendBtn, hdrName, isStreaming=false, streamingEl=null, sendHandler=null;
// FIX: track listeners to prevent leak on re-init
let _keyHandler=null, _clickHandler=null;
const MAX_MSGS = 120; // FIX: bound DOM growth (was unbounded => mem leak)

function init({ bubbleId="bubble", msgsId="msgs", inputId="chatInput", sendId="btnSend" }={}){
  bubble = document.getElementById(bubbleId);
  msgs = document.getElementById(msgsId);
  input = document.getElementById(inputId);
  sendBtn = document.getElementById(sendId);
  if(!bubble || !msgs || !input || !sendBtn) throw new Error("bubble elements missing");
  bubble.style.display = "none";
  bubble.classList.remove("open");
  // FIX: remove previous listeners if re-inited
  if (_keyHandler) try { input.removeEventListener("keydown", _keyHandler); } catch(e){}
  if (_clickHandler) try { sendBtn.removeEventListener("click", _clickHandler); } catch(e){}
  // Re-attach onInput handler if already set
  if (sendHandler) _bindHandlers();
}

function _bindHandlers(){
  _keyHandler = (e)=>{ if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); const v=input.value; input.value=""; if(sendHandler) sendHandler(v); }};
  _clickHandler = ()=>{ const v=input.value; input.value=""; if(sendHandler) sendHandler(v); };
  input.addEventListener("keydown", _keyHandler);
  sendBtn.addEventListener("click", _clickHandler);
}

function visible(){ return bubble.style.display !== "none"; }
function show(){
  bubble.style.display = "flex";
  bubble.classList.add("open");
  updateWindowSize();
  setTimeout(()=> input.focus(), 30);
}
function hide(){
  bubble.style.display = "none";
  bubble.classList.remove("open");
  updateWindowSize();
}
function toggle(){ visible() ? hide() : show(); }
function isVisible(){ return visible(); }

function _prune(){
  while (msgs.children.length > MAX_MSGS) msgs.removeChild(msgs.firstChild);
}

function append(who, text, kind){
  if(isStreaming && streamingEl){ finalizeStreaming(who, text); return; }
  const d = document.createElement("div");
  d.className = "msg " + (kind||"quillan");
  // FIX: use textContent for who + safe DOM creation to avoid innerHTML XSS even if esc fails
  const b = document.createElement("b");
  b.textContent = String(who)+": ";
  const span = document.createElement("span");
  // preserve line breaks without innerHTML
  const safe = String(text);
  // Use DocumentFragment to handle \n as <br> safely
  safe.split("\n").forEach((line, i)=>{
    if (i>0) span.appendChild(document.createElement("br"));
    span.appendChild(document.createTextNode(line));
  });
  d.appendChild(b); d.appendChild(span);
  msgs.appendChild(d);
  _prune();
  msgs.scrollTop = msgs.scrollHeight;
  show();
  return d;
}

function upsertStreaming(who, fullText){
  if(!isStreaming){
    streamingEl = document.createElement("div");
    streamingEl.className = "msg quillan streaming";
    const b = document.createElement("b"); b.textContent = String(who)+": ";
    const st = document.createElement("span"); st.className="streamText";
    String(fullText).split("\n").forEach((line,i)=>{ if(i>0) st.appendChild(document.createElement("br")); st.appendChild(document.createTextNode(line)); });
    const typing = document.createElement("span"); typing.className="typing";
    streamingEl.appendChild(b); streamingEl.appendChild(st); streamingEl.appendChild(typing);
    msgs.appendChild(streamingEl);
    show();
  } else if(streamingEl){
    const span = streamingEl.querySelector(".streamText");
    if(span){
      // rebuild safely
      span.textContent="";
      String(fullText).split("\n").forEach((line,i)=>{ if(i>0) span.appendChild(document.createElement("br")); span.appendChild(document.createTextNode(line)); });
    }
  } else {
    streamingEl = append(who, fullText, "quillan streaming");
    const t = document.createElement("span"); t.className="typing"; streamingEl.appendChild(t);
  }
  _prune();
  msgs.scrollTop = msgs.scrollHeight;
}

function finalizeStreaming(who, fullText){
  if(streamingEl){
    streamingEl.classList.remove("streaming");
    const span = streamingEl.querySelector(".streamText");
    if(span){
      span.textContent="";
      String(fullText).split("\n").forEach((line,i)=>{ if(i>0) span.appendChild(document.createElement("br")); span.appendChild(document.createTextNode(line)); });
    }
    const typing = streamingEl.querySelector(".typing");
    if(typing) typing.remove();
    streamingEl = null;
  } else if(fullText){
    append(who, fullText, "quillan");
  }
  msgs.scrollTop = msgs.scrollHeight;
}

function setStreaming(on){
  isStreaming = !!on;
  if(sendBtn) sendBtn.disabled = !!on;
  if(!on && streamingEl){
    const t = streamingEl.querySelector(".typing");
    if(t) t.remove();
  }
}

function clear(){ msgs.innerHTML=""; streamingEl=null; }
function onInput(handler){
  sendHandler = handler;
  // FIX: remove old before adding new
  if (_keyHandler) try { input.removeEventListener("keydown", _keyHandler); } catch(e){}
  if (_clickHandler) try { sendBtn.removeEventListener("click", _clickHandler); } catch(e){}
  _bindHandlers();
}
function getInput(){ return input.value; }
function clearInput(){ input.value = ""; }

function updateWindowSize(){
  // Use preload api if available, fallback to ipcRenderer
  try {
    const api = (typeof window !== "undefined" && window.api) ? window.api : null;
    if (api) { if(visible()) api.send("resize-window", 200+320+24, 360); else api.send("resize-window", 200+16, 200+60); return; }
  } catch(e){}
  try { const { ipcRenderer } = require("electron"); if(visible()) ipcRenderer.send("resize-window", 200+320+24, 360); else ipcRenderer.send("resize-window", 200+16, 200+60); } catch(e){}
}

function esc(s){
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function dispose(){
  if (_keyHandler) try { input.removeEventListener("keydown", _keyHandler); } catch(e){}
  if (_clickHandler) try { sendBtn.removeEventListener("click", _clickHandler); } catch(e){}
  _keyHandler=null; _clickHandler=null; sendHandler=null;
}

module.exports = { init, show, hide, toggle, isVisible, visible, append, clear, onInput, getInput, clearInput, upsertStreaming, finalizeStreaming, setStreaming, updateWindowSize, dispose };
