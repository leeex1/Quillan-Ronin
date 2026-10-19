(function () {
  const $ = (id) => document.getElementById(id);
  
  // StateManager for Decoupling
  const StateManager = {
    state: { mode: 'standard', history: [], busy: false },
    listeners: [],
    set(key, val) {
      this.state[key] = val;
      this.listeners.forEach(fn => fn(this.state));
    },
    get(key) { return this.state[key]; },
    subscribe(fn) { this.listeners.push(fn); }
  };

  const escapeHTML = (str) => String(str).replace(/[&<>"']/g, m => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[m]);

  const chatWindow = $('chatWindow');
  const input = $('userInput');
  const sendBtn = $('sendBtn');

  function scroll() { chatWindow.scrollTop = chatWindow.scrollHeight; }

  function addMsg(kind, content) {
    const d = document.createElement('div');
    d.className = 'message ' + kind;
    if (kind === 'user' || kind === 'system') {
      d.innerHTML = escapeHTML(content);
    } else if (kind === 'quillan' && window.marked) {
      d.innerHTML = marked.parse(content);
    } else {
      d.innerHTML = content;
    }
    chatWindow.appendChild(d);
    scroll();
    return d;
  }

  function setConn(on, label) {
    const dot = $('conn-dot'), lbl = $('conn-label');
    if (!dot || !lbl) return;
    dot.classList.toggle('on', on);
    lbl.textContent = label;
  }

  // API Service
  const ApiService = {
    async call(path, body) {
      try {
        const res = await fetch(path, {
          method: body ? 'POST' : 'GET',
          headers: { 'Content-Type': 'application/json' },
          body: body ? JSON.stringify(body) : undefined
        });
        const j = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(j.error || ('HTTP ' + res.status));
        return j;
      } catch (e) {
        console.error('API Error:', e);
        throw new Error('Connection failed or endpoint error.');
      }
    }
  };

  async function askQuillan(text) {
    if (StateManager.get('busy') || !text.trim()) return;
    
    StateManager.set('busy', true);
    sendBtn.disabled = true;
    addMsg('user', text);
    input.value = '';
    
    const steps = document.querySelectorAll('.wave-step');
    steps.forEach(s => s.classList.remove('active', 'done'));
    for (let i = 0; i < steps.length; i++) {
      steps[i].classList.add('active');
      await new Promise(r => setTimeout(r, 160 + Math.random() * 220));
      steps[i].classList.remove('active');
      steps[i].classList.add('done');
    }
    
    const typing = addMsg('quillan typing', '');
    try {
      const mode = StateManager.get('mode');
      const history = StateManager.get('history');
      const r = await ApiService.call('/api/chat', { message: text, mode, history: history.slice(-6) });
      
      typing.classList.remove('typing');
      typing.innerHTML = window.marked ? marked.parse(r.reply || '(empty)') : escapeHTML(r.reply || '(empty)');
      
      const newHistory = [...history, { role: 'user', content: text }, { role: 'assistant', content: r.reply }];
      StateManager.set('history', newHistory);
    } catch (e) {
      typing.classList.remove('typing');
      typing.innerHTML = '<span style="color:#ff3860">LINK ERROR:</span> ' + escapeHTML(e.message) +
        '<br><span class="small">Check server console.</span>';
    }
    
    StateManager.set('busy', false);
    sendBtn.disabled = false;
    scroll();
  }

  sendBtn.addEventListener('click', () => askQuillan(input.value));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askQuillan(input.value); }
  });

  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      StateManager.set('mode', btn.dataset.mode);
      addMsg('system', 'MODE SHIFT → ' + StateManager.get('mode').toUpperCase() + ' PROTOCOL ENGAGED');
    });
  });

  document.querySelectorAll('.tool-generate-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const pane = btn.closest('.tool-pane');
      const txt = (pane.querySelector('.tool-input').value || '').trim();
      if (!txt) return;
      if (pane.id === 'tool-text') { askQuillan(txt); pane.querySelector('.tool-input').value = ''; return; }
      if (pane.id !== 'tool-image') {
        addMsg('system', pane.id.toUpperCase() + ' GENERATION OFFLINE — TEXT + IMAGE PIPELINES ONLY IN THIS BUILD');
        return;
      }
      if (StateManager.get('busy')) return;
      
      StateManager.set('busy', true);
      addMsg('user', '🎨 ' + txt);
      const typing = addMsg('quillan typing', '');
      try {
        const r = await ApiService.call('/api/image', { prompt: txt });
        typing.classList.remove('typing');
        const imgUrl = escapeHTML(r.image); // basic sanitize
        typing.innerHTML = `<img src="${imgUrl}" style="max-width:100%;border-radius:8px;border:1px solid var(--line)">` +
          `<div class="small" style="margin-top:6px">SDXL-Turbo via local ComfyUI · saved to Comfy output/quillan/</div>`;
      } catch (e) {
        typing.classList.remove('typing');
        typing.innerHTML = '<span style="color:#ff3860">FORGE ERROR:</span> ' + escapeHTML(e.message);
      }
      StateManager.set('busy', false);
      scroll();
    });
  });

  setInterval(() => {
    const c = $('hud-clock');
    if (c) c.textContent = new Date().toTimeString().slice(0, 8);
  }, 1000);

  const jitter = () => {
    const busy = StateManager.get('busy');
    const load = $('val-load'), conf = $('val-conf'), en = $('val-energy'), sw = $('val-swarm');
    if (load) load.textContent = (busy ? 55 + Math.random() * 40 : 6 + Math.random() * 14).toFixed(0) + '%';
    if (conf) conf.textContent = (96.5 + Math.random() * 3.4).toFixed(1) + '%';
    if (en) en.textContent = (busy ? 4 + Math.random() * 3 : 1.8 + Math.random() * 1.2).toFixed(1) + 'e-8';
    if (sw && !sw.dataset.locked) sw.textContent = (224 + Math.random()).toFixed(0) + 'k';
  };
  setInterval(jitter, 2000); jitter();

  const cv = $('neuralCanvas');
  if (cv) {
    const ctx = cv.getContext('2d');
    const N = 32;
    const nodes = Array.from({ length: N }, (_, i) => ({
      x: Math.random(), y: Math.random(),
      vx: (Math.random() - .5) * .0016, vy: (Math.random() - .5) * .0016,
      tier: i % 4 === 0 ? 1 : 0
    }));
    
    let isVisible = true;
    let animFrame = null;
    
    const observer = new IntersectionObserver((entries) => {
      isVisible = entries[0].isIntersecting;
      if (isVisible && !animFrame) draw();
      else if (!isVisible && animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    });
    observer.observe(cv);

    function draw() {
      if (!isVisible) return;
      const w = cv.width = cv.clientWidth, h = cv.height = cv.clientHeight;
      ctx.clearRect(0, 0, w, h);
      nodes.forEach(n => {
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > 1) n.vx *= -1;
        if (n.y < 0 || n.y > 1) n.vy *= -1;
      });
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy;
        if (d2 < .02) {
          ctx.strokeStyle = `rgba(0,255,255,${(.16 * (1 - d2 / .02)).toFixed(3)})`;
          ctx.beginPath(); ctx.moveTo(a.x * w, a.y * h); ctx.lineTo(b.x * w, b.y * h); ctx.stroke();
        }
      }
      nodes.forEach(n => {
        ctx.fillStyle = n.tier ? '#e8b64c' : '#00ffff';
        ctx.shadowColor = n.tier ? '#e8b64c' : '#00ffff'; ctx.shadowBlur = 5;
        ctx.beginPath(); ctx.arc(n.x * w, n.y * h, n.tier ? 2.6 : 1.7, 0, 7); ctx.fill();
        ctx.shadowBlur = 0;
      });
      animFrame = requestAnimationFrame(draw);
    }
  }

  const orb = $('avatarContainer');
  if (orb && !orb.firstChild) orb.innerHTML = '<div class="core-orb"></div>';

  ApiService.call('/api/state').then(() => setConn(true, 'ONLINE')).catch(() => setConn(false, 'LOCAL ONLY'));
})();
