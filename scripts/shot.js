#!/usr/bin/env node
/* Screenshot one interactive in a chosen state, so a picture can be looked at
   rather than reasoned about.
   Usage: node scripts/shot.js <chapter-id> <figure-index> [--slider=0:1,1:0.5] [--click=2] [--wait=800] [--out=file.png] */
const fs = require('fs'); const path = require('path'); const http = require('http');
const { chromium } = require('playwright-core');
const ROOT = path.join(__dirname, '..');
const EXE = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const a = process.argv.slice(2);
const id = a[0], figIdx = parseInt(a[1] || '1', 10);
const arg = (n, d) => { const m = a.find(x => x.startsWith('--' + n + '=')); return m ? m.slice(n.length + 3) : d; };
(async () => {
  const srv = http.createServer((rq, rs) => {
    const p = path.join(ROOT, decodeURIComponent(rq.url.split('?')[0]).replace(/^\/+/, '') || 'index.html');
    fs.readFile(p, (e, d) => { if (e) { rs.writeHead(404); return rs.end(); } rs.writeHead(200, { 'content-type': MIME[path.extname(p)] || 'application/octet-stream', 'cache-control': 'no-store' }); rs.end(d); });
  });
  await new Promise(r => srv.listen(0, '127.0.0.1', r));
  const browser = await chromium.launch({ executablePath: EXE, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1400 }, deviceScaleFactor: 1 });
  await page.goto('http://127.0.0.1:' + srv.address().port + '/#/ch/' + id, { waitUntil: 'load' });
  await page.evaluate(i => { location.hash = '#/ch/' + i; }, id);
  await page.waitForTimeout(500);
  const fig = (await page.$$('.figure'))[figIdx - 1];
  if (!fig) { console.log('no figure ' + figIdx); await browser.close(); srv.close(); return; }
  const sl = arg('slider', '');
  if (sl) await page.evaluate(([i, spec]) => {
    const f = document.querySelectorAll('.figure')[i - 1];
    const rs = f.querySelectorAll('input[type=range]');
    spec.split(',').forEach(pair => {
      const [k, v] = pair.split(':'); const el = rs[+k]; if (!el) return;
      const lo = +el.min, hi = +el.max; el.value = String(lo + (+v) * (hi - lo));
      el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true }));
    });
  }, [figIdx, sl]);
  const ck = arg('click', '');
  if (ck) for (const c of ck.split(',')) await page.evaluate(([i, j]) => { const b = document.querySelectorAll('.figure')[i - 1].querySelectorAll('button')[+j]; if (b) b.click(); }, [figIdx, c]);
  await page.waitForTimeout(parseInt(arg('wait', '700'), 10));
  const out = arg('out', path.join(process.env.SHOT_DIR || '/tmp', id + '-fig' + figIdx + '.png'));
  fs.mkdirSync(path.dirname(out), { recursive: true });
  await fig.screenshot({ path: out });
  console.log(out);
  await browser.close(); srv.close();
})();
