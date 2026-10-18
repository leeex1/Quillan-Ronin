/*
  avatar.js — CLEAN: sprite visible immediately, GLB lazy on demand
  User requested GLB eventually but app was blank due to THREE import hang.
  Now: sprite shows instantly (purple procedural clippy). Click Quillan/JDXX button to lazy-load GLB via file:// import with timeout.
*/
let THREE = null; let GLTFLoader = null;
let mode = "sprite";
let state = "idle";
let t = 0;
let stageEl, canvas, ctx, mount3d, glowEl, onStateCb=null;
let scene, camera, renderer, loader, desk=null;
let avatars = {}; let current = "quillan"; let animId=null;
let spriteCfg = null; let sheetImg=null; let sheetReady=false;
let frame = 0, frameTick=0, lastTs=0;
let _disposed = false; // guard against running loop after dispose
let _pendingGLB = new Set(); // track pending GLB promises to abort

const BUILTIN = {
  quillan: { name:"quillan", type:"glb", sheet:null, frameW:200, frameH:200, fps:9, states:{ idle:{fps:6,frames:4,loop:true}, blink:{fps:14,frames:3,loop:false,next:"idle"}, talk:{fps:12,frames:4,loop:true}, think:{fps:5,frames:4,loop:true}, dance:{fps:14,frames:6,loop:true}, walk:{fps:12,frames:6,loop:true}, chill:{fps:4,frames:2,loop:true}, desk:{fps:6,frames:4,loop:true} } },
  jdxx:    { name:"jdxx",    type:"glb", sheet:null, frameW:200, frameH:200, fps:9, states:{ idle:{fps:7,frames:4,loop:true}, talk:{fps:12,frames:4,loop:true}, dance:{fps:16,frames:6,loop:true}, think:{fps:6,frames:3,loop:true}, walk:{fps:12,frames:6,loop:true}, desk:{fps:5,frames:2,loop:true} } },
  clippy:  { name:"clippy",  type:"sprite", sheet:null, frameW:200, frameH:200, fps:8, states:{ idle:{fps:6,frames:6,loop:true}, blink:{fps:18,frames:2,loop:false,next:"idle"}, talk:{fps:14,frames:4,loop:true}, think:{fps:4,frames:4,loop:true}, dance:{fps:15,frames:8,loop:true}, walk:{fps:13,frames:6,loop:true}, chill:{fps:3,frames:2,loop:true} } }
};

function log(m){ try{ console.log("[avatar] "+m); const fs=require("fs"); fs.appendFileSync("C:\\Users\\Admin\\AppData\\Local\\Temp\\opencode\\renderer.log", new Date().toISOString()+" [avatar] "+m+"\n"); }catch(e){} }

async function init({ stage, canvas: c, mount3d: m3, glowEl: g, onState }){
  // dispose previous loop if re-init (fixes RAF leak)
  if (animId) { try { cancelAnimationFrame(animId); } catch(e){} animId=null; }
  _disposed = false;
  stageEl=stage; canvas=c; mount3d=m3; glowEl=g; onStateCb=onState||null;
  ctx = canvas.getContext("2d");
  canvas.width=200; canvas.height=200;
  canvas.style.display="block";
  if(mount3d) mount3d.style.display="none";
  try{
    const fs=require("fs"); const p="C:\\02_QUILLAN\\quillan-app\\assets\\avatar\\config.json";
    if(fs.existsSync(p)){ const j=JSON.parse(fs.readFileSync(p,"utf-8")); log("config "+j.name+" current="+j.current); }
  }catch(e){ log("no config: "+e.message); }
  log("window.THREE check global="+(typeof window!=="undefined" && !!window.THREE)+" rev="+(window.THREE?window.THREE.REVISION:"none"));
  spriteCfg = BUILTIN["quillan"];
  lastTs = performance.now();
  if(animId) cancelAnimationFrame(animId);
  animId = requestAnimationFrame(loop);
  log("sprite loop started (visible) mode="+mode);
  log("avatar init done (sprite visible) mode="+mode+" current="+current+" — GLB lazy on avatar switch — alive ticker on");
  // FIX: store timeout so we can clear on dispose; don't auto-upgrade if already disposed
  const _autoTO = setTimeout(()=>{ if(!_disposed) show("quillan"); }, 800);
  // expose dispose cleanup
  if (typeof window !== "undefined") {
    window.addEventListener("beforeunload", dispose, { once:true });
    document.addEventListener("visibilitychange", ()=>{
      if (document.hidden) { if(animId){ cancelAnimationFrame(animId); animId=null; } }
      else if(!animId && !_disposed){ lastTs=performance.now(); animId=requestAnimationFrame(loop); }
    });
  }
  // auto-test removed
}

function loadSheet(url){
  // FIX: dispose previous image to prevent img mem leak
  if (sheetImg) { try { sheetImg.onload=null; sheetImg.onerror=null; sheetImg.src=""; } catch(e){} sheetImg=null; sheetReady=false; }
  sheetImg = new Image();
  sheetImg.onload = ()=>{ sheetReady=true; log("sheet ready "+url); };
  sheetImg.onerror = ()=>{ sheetReady=false; log("sheet missing "+url); try{ sheetImg.src=""; }catch(e){} sheetImg=null; };
  sheetImg.src = url;
}

async function ensureThree(){
  if(THREE) return THREE;
  // Try global THREE from <script> tag first (three@0.128 UMD)
  if(typeof window !== "undefined" && window.THREE){
    THREE = window.THREE;
    log("THREE global r"+(THREE.REVISION||"?")+" from script tag");
    if(window.THREE.GLTFLoader) { GLTFLoader = window.THREE.GLTFLoader; log("GLTFLoader global from script tag"); }
    else if(THREE.GLTFLoader) { GLTFLoader = THREE.GLTFLoader; log("GLTFLoader from THREE.GLTFLoader"); }
    if(GLTFLoader) return THREE;
  }
  log("global THREE not found, trying require...");
  try{ THREE = require("three"); log("THREE require r"+(THREE.REVISION||"?")+" ok"); }catch(e){ log("require fail "+e.message); }
  if(THREE && THREE.GLTFLoader) { GLTFLoader = THREE.GLTFLoader; log("GLTFLoader from require"); }
  // If still no GLTFLoader, try dynamic import as last resort
  if(!GLTFLoader){
    const tryImport = (url, label, ms=2500) => Promise.race([
      import(url).then(m=>{ log(label+" ok"); return m; }),
      new Promise((_,rej)=> setTimeout(()=> rej(new Error(label+" timeout")), ms))
    ]);
    try{
      const lm = await tryImport("three/examples/jsm/loaders/GLTFLoader.js", "GLTF bare", 2500);
      GLTFLoader = lm.GLTFLoader; log("GLTFLoader bare import ok");
    }catch(e){ log("GLTF bare fail "+e.message); try{ const lm2 = await tryImport("file:///C:/02_QUILLAN/quillan-app/node_modules/three/examples/jsm/loaders/GLTFLoader.js", "GLTF file", 3000); GLTFLoader=lm2.GLTFLoader; log("GLTF file ok"); }catch(e2){ log("GLTF failed both "+e2.message); }}
  }
  if(!GLTFLoader) log("WARNING: GLTFLoader still missing — will show capsule fallback");
  return THREE;
}

async function loadGLBs(){
  if(!THREE) throw new Error("no THREE");
  if(scene) return;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100); camera.position.set(0, 1.05, 3.4); camera.lookAt(0, 0.35, 0);
  renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true, powerPreference:"low-power" });
  // FIX: limit pixelRatio to 1.5 to cap GPU mem (was devicePixelRatio unlimited -> 3x VRAM on retina)
  renderer.setSize(200,200); renderer.setPixelRatio(Math.min(window.devicePixelRatio||1, 1.5)); renderer.shadowMap.enabled=true;
  if(THREE.SRGBColorSpace) renderer.outputColorSpace = THREE.SRGBColorSpace;
  // FIX: dispose old renderer before clearing mount
  if (mount3d.firstChild && mount3d.firstChild !== renderer.domElement) {
    try { mount3d.innerHTML=""; } catch(e){}
  } else mount3d.innerHTML="";
  mount3d.appendChild(renderer.domElement);
  const _wheelFn = (e)=>{ e.preventDefault(); camera.position.z += e.deltaY*0.004; camera.position.z = Math.max(1.8, Math.min(6, camera.position.z)); camera.lookAt(0,0.35,0); };
  renderer.domElement._wheelFn=_wheelFn;
  renderer.domElement.addEventListener("wheel", _wheelFn, {passive:false});
  log("renderer 200x200 alpha");
  const hemi=new THREE.HemisphereLight(0xffffff,0x1a1a2e,0.7); scene.add(hemi);
  const key=new THREE.DirectionalLight(0xffffff,1.1); key.position.set(2,3,2); key.castShadow=true; scene.add(key);
  const rim=new THREE.DirectionalLight(0x7C3AED,0.7); rim.position.set(-2,1,-2); scene.add(rim);
  const ground=new THREE.Mesh(new THREE.CircleGeometry(1.6,32), new THREE.MeshStandardMaterial({color:0x1a1a2e,roughness:0.92})); ground.rotation.x=-Math.PI/2; ground.position.y=-0.12; ground.receiveShadow=true; scene.add(ground);
  let fbGeo; if(THREE.CapsuleGeometry) fbGeo=new THREE.CapsuleGeometry(0.32,0.9,8,16); else fbGeo=new THREE.BoxGeometry(0.64,1.2,0.64); const fb=new THREE.Mesh(fbGeo, new THREE.MeshStandardMaterial({color:0x7C3AED,roughness:0.5})); fb.position.y=0.35; fb.name="fallback"; scene.add(fb); avatars["fallback"]={mesh:fb,baseY:0.35,baseRotY:0};
  log("fallback added");
  if(!GLTFLoader){ log("no GLTFLoader — fallback only — window.THREE.GLTFLoader="+(typeof window!=="undefined" && !!(window.THREE&&window.THREE.GLTFLoader))); return; }
  loader=new GLTFLoader();
  Promise.allSettled([ tryImport("quillan","assets/avatar/quillan_textured.glb",-0.14), tryImport("jdxx","assets/avatar/jdxx_textured.glb",0.14) ]).then(r=> log("GLB settled "+JSON.stringify(r.map(x=>x.status))));
  const deskGeo=new THREE.BoxGeometry(1.1,0.06,0.55); const deskMat=new THREE.MeshStandardMaterial({color:0x2a1f18,roughness:0.65}); desk=new THREE.Mesh(deskGeo,deskMat); desk.position.set(0,0.52,0.35); desk.castShadow=true; desk.receiveShadow=true; desk.visible=false; scene.add(desk);
  const legM=new THREE.BoxGeometry(0.06,0.52,0.06); const l1=new THREE.Mesh(legM,deskMat); l1.position.set(-0.45,0.26,0.15); desk.add(l1); const l2=new THREE.Mesh(legM,deskMat); l2.position.set(0.45,0.26,0.15); desk.add(l2);
  log("desk ready");
}

async function tryImport(name, p, rotY){
  try{
    log("loading "+p+" ...");
    const gltf=await loader.loadAsync(p);
    const m=gltf.scene; m.traverse(o=>{ if(o.isMesh){o.castShadow=true;o.receiveShadow=true;}});
    if(avatars["fallback"]) avatars["fallback"].mesh.visible=false;
    scene.add(m);
    const box=new THREE.Box3().setFromObject(m); const center=box.getCenter(new THREE.Vector3()); const size=box.getSize(new THREE.Vector3());
    const s=1.35/Math.max(size.x,size.y,size.z); m.scale.setScalar(s);
    m.position.set(-center.x*s, -center.y*s, -center.z*s); m.position.y += 0.35; m.rotation.y=rotY; log(name+" bbox "+center.x.toFixed(2)+","+center.y.toFixed(2)+" size "+size.x.toFixed(2));
    avatars[name]={mesh:m,baseY:m.position.y,baseRotY:rotY,size}; log(name+" GLB ready "+s.toFixed(2));
    if(name===current) show(name);
    return m;
  }catch(e){ log(name+" GLB fail "+e.message); throw e; }
}

let _firstShow=true;
function show(name, {forceGLB=false}={}){
  // On first boot, force sprite to keep window visible — GLB loads only on explicit button
  const isFirst = _firstShow; _firstShow=false;
  if(isFirst && (name==="quillan"||name==="jdxx")){
    // Show sprite version of quillan/jdxx on first boot, don't block
    const savedMode=mode;
    mode="sprite"; canvas.style.display="block"; if(mount3d) mount3d.style.display="none";
    // fall through to sprite path below, but remember to allow lazy later
    // Do not trigger ensureThree yet
  }
  current=name;
  if(BUILTIN[name]) spriteCfg=BUILTIN[name]; else spriteCfg=BUILTIN.quillan;
  frame=0; frameTick=0;
  const wantGLB = (name==="quillan"||name==="jdxx") && !isFirst;
  if(wantGLB){
    if(!THREE || !scene){
      log("show "+name+" — lazy loading 3D...");
      ensureThree().then(()=> loadGLBs().then(()=> {
        // switch to three after load
        mode="three"; canvas.style.display="none"; if(mount3d) mount3d.style.display="block";
        if(avatars[name]){ for(const [k,v] of Object.entries(avatars)) v.mesh.visible=(k===name); }
        log("switched to THREE for "+name);
      })).catch(e=>{ log("lazy 3D fail "+e.message+" staying sprite"); mode="sprite"; canvas.style.display="block"; if(mount3d) mount3d.style.display="none"; });
      // keep sprite visible meanwhile
      canvas.style.display="block"; if(mount3d) mount3d.style.display="none";
      document.querySelectorAll(".avatarBtn").forEach(b=> b.classList.toggle("active", b.dataset.avatar===name));
      return;
    } else {
      mode="three"; canvas.style.display="none"; if(mount3d) mount3d.style.display="block";
      for(const [k,v] of Object.entries(avatars)) v.mesh.visible=(k===name);
      log("show "+name+" mode=three (already loaded)");
    }
  } else {
    mode="sprite"; canvas.style.display="block"; if(mount3d) mount3d.style.display="none";
    try{ const fs=require("fs"); const cand="C:\\02_QUILLAN\\quillan-app\\assets\\avatar\\"+name+"_sheet.png"; if(fs.existsSync(cand)) loadSheet("assets/avatar/"+name+"_sheet.png"); else {sheetImg=null; sheetReady=false;} }catch(e){}
    log("show "+name+" mode=sprite");
  }
  document.querySelectorAll(".avatarBtn").forEach(b=> b.classList.toggle("active", b.dataset.avatar===name));
}

function setState(s){
  if(!s) return;
  const prev=state; state=s; frame=0; frameTick=0; t=0;
  if(desk){ if(s==="desk") desk.visible=true; else if(s==="idle"||s==="chill") desk.visible=false; }
  if(glowEl) glowEl.className="glow "+(s==="think"?"think":s==="talk"||s==="stream"?"talk":s==="dance"?"dance":s==="idle"?"":"on");
  if(onStateCb) onStateCb(s);
  if(prev!==s) log("state "+prev+" -> "+s);
}
function getState(){ return state; }
function blink(){ if(state==="idle"){ setState("blink"); setTimeout(()=> { if(state==="blink") setState("idle"); }, 420); }}

function loop(ts){
  if (_disposed) return;
  animId=requestAnimationFrame(loop);
  const dt = Math.min(0.05, (ts - lastTs)/1000 || 0.016); lastTs=ts; t+=dt;
  // FIX: skip rendering when page hidden to save GPU mem (cooperates with main suspend)
  if (typeof document !== "undefined" && document.hidden) return;
  if(mode==="sprite"){
    const stCfg = (spriteCfg && spriteCfg.states && spriteCfg.states[state]) || spriteCfg.states["idle"] || {fps:7,frames:4};
    const fps = stCfg.fps||7; frameTick += dt*fps;
    if(frameTick>=1){ const inc=Math.floor(frameTick); frameTick-=inc; frame+=inc; const total=stCfg.frames||4; if(frame>=total){ if(stCfg.loop===false){ frame=total-1; if(stCfg.next) setState(stCfg.next); } else frame%=total; }}
    if(sheetReady && sheetImg) drawSheetFrame(stCfg, frame); else drawProcedural(state, frame, t);
  }
  if(mode==="three" && scene && renderer){
    const entry=avatars[current] || avatars["fallback"]; const m=entry && entry.mesh;
    if(m){
      const baseY=entry.baseY, baseRotY=entry.baseRotY;
      if(state==="idle"){ m.position.y=baseY+Math.sin(t*1.4)*0.035 + Math.sin(t*1.8)*0.015; m.rotation.y=baseRotY+Math.sin(t*0.45)*0.09; }
      else if(state==="dance"){ const b=Math.abs(Math.sin(t*7.2)); m.position.y=baseY+b*0.14; m.rotation.y=baseRotY+Math.sin(t*6)*0.35; m.rotation.z=Math.sin(t*7)*0.22; }
      else if(state==="walk"){ m.position.y=baseY+Math.abs(Math.sin(t*9))*0.035; }
      else if(state==="chill"){ m.position.y=baseY-0.12; }
      else if(state==="think"){ m.rotation.z=Math.sin(t*3.2)*0.05; }
      else if(state==="talk"||state==="stream"){ m.position.y=baseY+Math.abs(Math.sin(t*8))*0.025; }
    }
    for(const [k,v] of Object.entries(avatars)){ if(v.mesh!==m && !v.mesh.visible){ v.mesh.position.y = v.baseY + Math.sin(t*1.2)*0.01; } }
    renderer.render(scene,camera);
  }
}

function dispose(){
  _disposed = true;
  if (animId) { try{ cancelAnimationFrame(animId);}catch(e){} animId=null; }
  if (sheetImg) { try{ sheetImg.onload=null; sheetImg.onerror=null; sheetImg.src=""; }catch(e){} sheetImg=null; }
  // Dispose Three resources
  try {
    if (renderer) {
      if (renderer.domElement && renderer.domElement._wheelFn) renderer.domElement.removeEventListener("wheel", renderer.domElement._wheelFn);
      renderer.dispose();
      // dispose geometries/materials
      if (scene) scene.traverse(o=>{ if(o.geometry) try{o.geometry.dispose()}catch(e){}; if(o.material){ const mats=Array.isArray(o.material)?o.material:[o.material]; mats.forEach(m=>{ try{ if(m.map) m.map.dispose(); m.dispose(); }catch(e){} }); } });
      if (mount3d && renderer.domElement && mount3d.contains(renderer.domElement)) mount3d.removeChild(renderer.domElement);
    }
  } catch(e){ log("dispose fail "+e.message); }
  scene=null; camera=null; renderer=null; loader=null;
  log("avatar disposed");
}

function drawSheetFrame(stCfg, f){
  ctx.clearRect(0,0,200,200);
  const fw=Math.floor(sheetImg.width/(stCfg.frames||4)), fh=sheetImg.height, sx=f*fw, sy=0, sw=fw, sh=fh;
  const scale=Math.min(200/fw,200/fh)*0.86, dw=fw*scale, dh=fh*scale, dx=(200-dw)/2, dy=(200-dh)/2+6;
  ctx.imageSmoothingEnabled=true; ctx.drawImage(sheetImg,sx,sy,sw,sh,dx,dy,dw,dh);
  ctx.fillStyle="rgba(0,0,0,.18)"; ctx.beginPath(); ctx.ellipse(100,182,36,8,0,0,Math.PI*2); ctx.fill();
}
function drawProcedural(s, f, time){
  ctx.clearRect(0,0,200,200);
  ctx.fillStyle="rgba(0,0,0,.18)"; ctx.beginPath(); ctx.ellipse(100,182,36,8,0,0,Math.PI*2); ctx.fill();
  const colors = current==="jdxx" ? {body:"#ff6b35"} : current==="clippy" ? {body:"#c0c0be"} : {body:"#7C3AED"};
  let bob=0, tilt=0, scale=1, eyeSquint=0, mouth=0;
  if(s==="idle"){ bob=Math.sin(time*1.4)*3; tilt=Math.sin(time*0.45)*0.05; eyeSquint=(f===2?0.72:0); }
  else if(s==="blink"){ eyeSquint=f===1?0.95:f===2?0.2:0; }
  else if(s==="talk"){ bob=Math.abs(Math.sin(time*8))*4; mouth=(f%2===0?1:0); }
  else if(s==="think"){ tilt=Math.sin(time*3.2)*0.06; }
  else if(s==="dance"){ bob=Math.abs(Math.sin(time*7.2))*10; tilt=Math.sin(time*6)*0.35; scale=1+Math.abs(Math.sin(time*7.2))*0.05; }
  else if(s==="walk"){ bob=Math.abs(Math.sin(time*9))*5; }
  else if(s==="chill"){ bob=-8; tilt=0.14; }
  else if(s==="stream"){ bob=Math.sin(time*6)*1.2; mouth=(frame%2); }
  ctx.save(); ctx.translate(100,92+bob); ctx.rotate(tilt); ctx.scale(scale,scale);
  ctx.fillStyle=colors.body; ctx.strokeStyle="rgba(0,0,0,.22)"; ctx.lineWidth=2; roundRect(ctx,-44,-54,88,96,22); ctx.fill(); ctx.stroke();
  ctx.fillStyle="rgba(255,255,255,.96)"; roundRect(ctx,-34,-42,68,54,12); ctx.fill();
  const eyeY=-18, eyeGap=22, eyeR=10;
  [-eyeGap/2,eyeGap/2].forEach((ex,i)=>{
    ctx.fillStyle="#1a1a2e"; ctx.beginPath(); if(eyeSquint>0.05){ const h=eyeR*2*(1-eyeSquint*0.9); ctx.ellipse(ex,eyeY,eyeR,Math.max(1.5,h/2),0,0,Math.PI*2); } else ctx.arc(ex,eyeY,eyeR,0,Math.PI*2); ctx.fill();
    if(eyeSquint<0.6){ const px=ex+Math.sin(time*0.7+i)*1.2; ctx.fillStyle="#111"; ctx.beginPath(); ctx.arc(px,eyeY,eyeSquint>0?2.2:3.4,0,Math.PI*2); ctx.fill(); ctx.fillStyle="rgba(255,255,255,.92)"; ctx.beginPath(); ctx.arc(px-1.4,eyeY-1.4,1.2,0,Math.PI*2); ctx.fill(); }
  });
  ctx.strokeStyle="rgba(0,0,0,.62)"; ctx.lineWidth=1.8; ctx.beginPath();
  if(mouth===1||s==="talk"||s==="stream"){ ctx.fillStyle="rgba(0,0,0,.78)"; ctx.beginPath(); ctx.ellipse(0,8,7,6,0,0,Math.PI*2); ctx.fill(); }
  else { ctx.beginPath(); ctx.arc(0,10,6,0.2*Math.PI,0.8*Math.PI); ctx.stroke(); }
  ctx.restore();
  if(s==="desk"){ ctx.fillStyle="rgba(42,31,24,.96)"; roundRect(ctx,22,112,108,9,3); ctx.fill(); }
  ctx.fillStyle="rgba(255,255,255,.72)"; ctx.font="600 8px Consolas, monospace"; ctx.textAlign="center"; ctx.fillText(current.toUpperCase()+" · "+s.toUpperCase(),100,196);
}
function roundRect(c,x,y,w,h,r){ c.beginPath(); c.moveTo(x+r,y); c.arcTo(x+w,y,x+w,y+h,r); c.arcTo(x+w,y+h,x,y+h,r); c.arcTo(x,y+h,x,y,r); c.arcTo(x,y,x+w,y,r); c.closePath(); }
function setSpriteSheet(url){ loadSheet(url); mode="sprite"; canvas.style.display="block"; if(mount3d) mount3d.style.display="none"; }
function setSpriteConfig(cfg){ spriteCfg=Object.assign({}, spriteCfg||BUILTIN.quillan, cfg); if(cfg.sheet) loadSheet(cfg.sheet); frame=0; frameTick=0; }
function getConfig(){ return spriteCfg; }
module.exports = { init, show, setState, getState, blink, setSpriteSheet, setSpriteConfig, getConfig, ensureThree, loadGLBs, dispose };













