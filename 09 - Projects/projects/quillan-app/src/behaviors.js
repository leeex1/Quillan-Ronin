let wanderTimer=null, nudgeTimer=null, blinkInterval=null;
let _stopped=false;

function startBehaviors(avatar, bubble, { onProactive=()=>{} }={}){
  _stopped=false;
  // clear any previous (fixes re-init leak)
  stop();
  function scheduleWander(){
    if (_stopped) return;
    const delay = 38000 + Math.random()*34000;
    wanderTimer = setTimeout(()=>{
      if(_stopped) return;
      if(avatar.getState()==="idle" && !bubble.isVisible()){
        avatar.setState("walk");
        // walk lasts ~4.2s then back to idle — store timeout so we can clear on stop
        const t2=setTimeout(()=> { if(avatar.getState()==="walk") avatar.setState("idle"); }, 4300);
        // if stopped before walk ends, clear
        if (_stopped) clearTimeout(t2);
      }
      if(!_stopped) scheduleWander();
    }, delay);
  }

  const nudges = [
    "It looks like you're coding — need a hand?",
    "Psst — I can watch your files and remind you to commit.",
    "Want me to keep your window tidy? Click desk to focus.",
    "I'm here if you need a rubber duck."
  ];
  function scheduleNudge(){
    if (_stopped) return;
    nudgeTimer = setTimeout(()=>{
      if(_stopped) return;
      if(!bubble.isVisible() && avatar.getState()==="idle" && Math.random()<0.45){
        const msg = nudges[Math.floor(Math.random()*nudges.length)];
        avatar.setState("think");
        setTimeout(()=> { if(!_stopped && avatar.getState()==="think") avatar.setState("idle"); }, 900);
        onProactive(msg, 5000);
      }
      if(!_stopped) scheduleNudge();
    }, 110000 + Math.random()*90000);
  }

  scheduleWander();
  scheduleNudge();

  // FIX: store interval id so it can be cleared (was anonymous leak)
  blinkInterval = setInterval(()=>{
    if(_stopped) return;
    if(avatar.getState()==="idle" && Math.random()<0.18){
      avatar.blink && avatar.blink();
    }
  }, 2600);
}

function stop(){
  _stopped=true;
  if(wanderTimer) { clearTimeout(wanderTimer); wanderTimer=null; }
  if(nudgeTimer) { clearTimeout(nudgeTimer); nudgeTimer=null; }
  if(blinkInterval) { clearInterval(blinkInterval); blinkInterval=null; }
}

module.exports = { startBehaviors, stop };
