const fs=require("fs");
const https=require("https");
https.get("https://leeex1.github.io/Quillan-Ronin/", res=>{
  let d=""; res.on("data",c=>d+=c); res.on("end",()=>{
    console.log("len",d.length);
    // Extract filterAlbum
    const m=d.match(/function filterAlbum[\s\S]*?reindexVisibleTracks\(\);\s*}/);
    console.log(m ? m[0].slice(0,500) : "not found");
    // Check for syntax error by trying to parse script
    const scriptCount=(d.match(/<script/g)||[]).length;
    console.log("scripts",scriptCount);
    // Try to find JDXX rows
    const jdxx=(d.match(/data-album="JDXX Rebooted"/g)||[]).length;
    console.log("jdxx",jdxx);
    // Try to eval filterAlbum in vm
    const vm=require("vm");
    const code=m ? m[0] : "";
    try { vm.runInNewContext(code); console.log("filter parses ok"); } catch(e){ console.log("parse err",e.message); }
  });
});
