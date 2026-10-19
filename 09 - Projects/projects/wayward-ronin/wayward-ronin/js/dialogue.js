// dialogue.js — data-driven dialogue trees with typewriter text and
// keyboard/mouse choice selection. Trees are defined in story.js.

export class Dialogue {
  constructor(audio) {
    this.audio = audio;
    this.panel = document.getElementById('dialogue');
    this.nameEl = document.getElementById('dlg-name');
    this.textEl = document.getElementById('dlg-text');
    this.choicesEl = document.getElementById('dlg-choices');
    this.contEl = document.getElementById('dlg-continue');
    this.active = false;
    this.tree = null;
    this.nodeId = null;
    this.ctx = null;
    this.onEnd = null;
    this.typing = false;
    this.fullText = '';
    this.typeTimer = null;
    this.waitingChoice = false;

    this.textEl.addEventListener('click', () => this.advance());
    this.panel.addEventListener('click', (e) => {
      if (e.target === this.panel) this.advance();
    });
    window.addEventListener('keydown', (e) => {
      if (!this.active) return;
      if (this.waitingChoice && ['1', '2', '3', '4'].includes(e.key)) {
        const i = parseInt(e.key, 10) - 1;
        const btns = this.choicesEl.children;
        if (btns[i]) btns[i].click();
      }
    });
  }

  get isActive() { return this.active; }

  start(tree, ctx, onEnd) {
    this.tree = tree; this.ctx = ctx; this.onEnd = onEnd || null;
    this.active = true;
    this.panel.classList.remove('hidden');
    this.showNode(tree.start);
  }

  showNode(id) {
    const node = this.tree.nodes[id];
    if (!node) { this.end(); return; }
    this.nodeId = id;
    if (node.onEnter) node.onEnter(this.ctx);
    const text = typeof node.text === 'function' ? node.text(this.ctx) : node.text;
    this.nameEl.textContent = node.speaker || this.tree.speaker || '';
    this.nameEl.style.color = this.tree.plate || '#d4a24e';
    this.choicesEl.innerHTML = '';
    this.waitingChoice = false;
    this.typeText(text, () => this.showChoices(node));
  }

  typeText(text, done) {
    this.fullText = text;
    this.textEl.textContent = '';
    this.typing = true;
    this.contEl.classList.add('hidden');
    let i = 0;
    let blip = 0;
    clearInterval(this.typeTimer);
    this.typeTimer = setInterval(() => {
      i += 2;
      this.textEl.textContent = text.slice(0, i);
      if (++blip % 4 === 0) this.audio.dialogueBlip();
      if (i >= text.length) {
        clearInterval(this.typeTimer);
        this.typing = false;
        done();
      }
    }, 18);
  }

  showChoices(node) {
    const avail = (node.choices || []).filter(c => !c.if || c.if(this.ctx));
    if (avail.length === 0) {
      this.contEl.classList.remove('hidden');
      return;
    }
    this.waitingChoice = true;
    avail.forEach((c, i) => {
      const b = document.createElement('button');
      b.className = 'dlg-choice';
      b.innerHTML = `<span class="key">${i + 1}</span>${c.text}`;
      b.addEventListener('click', (e) => {
        e.stopPropagation();
        this.audio.choice();
        if (c.do) c.do(this.ctx);
        if (c.to) this.showNode(c.to);
        else this.end();
      });
      this.choicesEl.appendChild(b);
    });
  }

  // E key or click: finish typing, otherwise advance/close.
  advance() {
    if (!this.active) return;
    if (this.typing) {
      clearInterval(this.typeTimer);
      this.typing = false;
      this.textEl.textContent = this.fullText;
      this.showChoices(this.tree.nodes[this.nodeId]);
      return;
    }
    if (this.waitingChoice) return;
    const node = this.tree.nodes[this.nodeId];
    if (node && node.to) this.showNode(node.to); // linear chain node
    else this.end();
  }

  // Programmatic choice (touch buttons / tests). Clicks the i-th choice button.
  choose(i) {
    if (!this.active || !this.waitingChoice) return false;
    const btns = this.choicesEl.children;
    if (btns[i]) { btns[i].click(); return true; }
    return false;
  }

  end() {
    clearInterval(this.typeTimer);
    this.active = false;
    this.panel.classList.add('hidden');
    this.choicesEl.innerHTML = '';
    const cb = this.onEnd;
    this.onEnd = null;
    if (cb) cb();
  }
}
