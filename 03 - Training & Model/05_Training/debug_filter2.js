const https=require("https");
const {JSDOM}=require("jsdom");
https.get("https://leeex1.github.io/Quillan-Ronin/", res=>{
  let d=""; res.on("data",c=>d+=c); res.on("end",()=>{
    const dom=new JSDOM(d, {runScripts:"dangerously", resources:"usable"});
    const win=dom.window;
    setTimeout(()=>{
      try {
        win.filterAlbum("JDXX Rebooted");
        const rows=win.document.querySelectorAll(".audio-track-row");
        let visible=0; rows.forEach(r=>{ if(r.style.display!=="none") visible++; });
        console.log("after JDXX filter visible",visible);
        const jdxxVisible=win.document.querySelectorAll(".audio-track-row[data-album=\"JDXX Rebooted\"]");
        let v2=0; jdxxVisible.forEach(r=>{ if(r.style.display!=="none") v2++; });
        console.log("jdxx visible",v2, "total jdxx",jdxxVisible.length);
        // Check style of first jdxx row
        if(jdxxVisible[0]) console.log("first style", jdxxVisible[0].style.display, jdxxVisible[0].getAttribute("data-album"));
      } catch(e){ console.log("err",e.message, e.stack); }
    },2000);
  });
});
