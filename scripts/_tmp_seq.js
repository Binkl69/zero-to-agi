/* click/slide an interactive in an arbitrary ORDER: --seq=c0,c1,s1:1  (c=button idx, s=range idx:frac) */
const fs=require('fs'),path=require('path'),http=require('http');
const {chromium}=require('playwright-core');
const ROOT='/home/user/zero-to-agi';
const EXE='/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const MIME={'.html':'text/html','.js':'text/javascript','.css':'text/css'};
const a=process.argv.slice(2); const id=a[0], figIdx=parseInt(a[1],10);
const arg=(n,d)=>{const m=a.find(x=>x.startsWith('--'+n+'='));return m?m.slice(n.length+3):d;};
(async()=>{
 const srv=http.createServer((rq,rs)=>{const p=path.join(ROOT,decodeURIComponent(rq.url.split('?')[0]).replace(/^\/+/,'')||'index.html');
  fs.readFile(p,(e,d)=>{if(e){rs.writeHead(404);return rs.end();}rs.writeHead(200,{'content-type':MIME[path.extname(p)]||'application/octet-stream','cache-control':'no-store'});rs.end(d);});});
 await new Promise(r=>srv.listen(0,'127.0.0.1',r));
 const browser=await chromium.launch({executablePath:EXE,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1280,height:1400},deviceScaleFactor:1});
 await page.goto('http://127.0.0.1:'+srv.address().port+'/#/ch/'+id,{waitUntil:'load'});
 await page.evaluate(i=>{location.hash='#/ch/'+i;},id); await page.waitForTimeout(600);
 for(const step of (arg('seq','')||'').split(',').filter(Boolean)){
   if(step[0]==='c') await page.evaluate(([i,j])=>{const b=document.querySelectorAll('.figure')[i-1].querySelectorAll('button')[+j];if(b)b.click();},[figIdx,step.slice(1)]);
   else { const [k,v]=step.slice(1).split(':');
     await page.evaluate(([i,k,v])=>{const el=document.querySelectorAll('.figure')[i-1].querySelectorAll('input[type=range]')[+k];
       if(!el)return; const lo=+el.min,hi=+el.max; el.value=String(lo+(+v)*(hi-lo));
       el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));},[figIdx,k,v]); }
   await page.waitForTimeout(250);
 }
 await page.waitForTimeout(parseInt(arg('wait','700'),10));
 const out=arg('out','/tmp/ch10/seq.png');
 await (await page.$$('.figure'))[figIdx-1].screenshot({path:out});
 console.log(out); await browser.close(); srv.close();
})();
