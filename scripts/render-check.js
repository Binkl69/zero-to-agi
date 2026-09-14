#!/usr/bin/env node
/* Rendering check in a real browser.
   Usage: node scripts/render-check.js [chapter-id ...]   [--shots]

   paint.js runs the same checks against a modelled canvas, which is fast and
   good enough for a pre-commit loop but has to approximate text widths. This
   loads the actual page in Chromium, instruments the real
   CanvasRenderingContext2D, and runs the checks on measurements the browser
   itself produced. It is the one that decides. With --shots it also writes a PNG
   of every interactive so the pictures can be looked at rather than inferred. */
const fs = require('fs');
const path = require('path');
const http = require('http');
const { chromium } = require('playwright-core');
const instrumentSource = require('./lib/instrument.js');

const ROOT = path.join(__dirname, '..');
const EXE = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const args = process.argv.slice(2);
const SHOTS = args.includes('--shots');
const only = args.filter(a => !a.startsWith('--'));
const SHOT_DIR = process.env.SHOT_DIR || path.join(ROOT, '.render-shots');

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png' };
function serve() {
  return new Promise(res => {
    const srv = http.createServer((req, rq) => {
      const p = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '') || 'index.html');
      if (!p.startsWith(ROOT)) { rq.writeHead(403); return rq.end(); }
      fs.readFile(p, (e, d) => {
        if (e) { rq.writeHead(404); return rq.end('nope'); }
        rq.writeHead(200, { 'content-type': MIME[path.extname(p)] || 'application/octet-stream', 'cache-control': 'no-store' });
        rq.end(d);
      });
    });
    srv.listen(0, '127.0.0.1', () => res(srv));
  });
}

(async () => {
  const srv = await serve();
  const port = srv.address().port;
  const browser = await chromium.launch({ executablePath: EXE, args: ['--no-sandbox', '--font-render-hinting=none'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1400 }, deviceScaleFactor: 1 });
  await page.addInitScript(instrumentSource());
  await page.addInitScript(fs.readFileSync(path.join(__dirname, 'lib/checks.js'), 'utf8')
    .replace(/^module\.exports[\s\S]*$/m, '') + '\nwindow.__ZTA_CHECK = checkFrame;');

  const ids = fs.readdirSync(path.join(ROOT, 'app/chapters')).filter(f => f.endsWith('.js')).map(f => f.replace(/\.js$/, '')).sort()
    .filter(id => !only.length || only.some(o => id.includes(o)));

  if (SHOTS) fs.mkdirSync(SHOT_DIR, { recursive: true });
  let total = 0, bad = 0;
  const consoleErrors = [];
  page.on('pageerror', e => consoleErrors.push(String(e.message)));

  for (const id of ids) {
    consoleErrors.length = 0;
    await page.goto('http://127.0.0.1:' + port + '/#/ch/' + id, { waitUntil: 'load' });
    await page.evaluate((i) => { location.hash = '#/ch/' + i; }, id);
    await page.waitForTimeout(400);
    await page.waitForFunction(() => document.querySelectorAll('.figure canvas').length > 0, null, { timeout: 15000 }).catch(() => {});

    /* which canvas is which: label by the figure's caption so a report points
       at something the author can find */
    const labels = await page.evaluate(() => {
      const out = [];
      document.querySelectorAll('canvas').forEach((cv, i) => {
        let f = cv.closest('.figure');
        let cap = f && f.querySelector('.figure-caption');
        out.push({ i, cap: cap ? cap.textContent.trim().slice(0, 70) : '(no caption)' });
      });
      return out;
    });

    const seen = new Map();
    const capture = async (scene) => {
      const frames = await page.evaluate(async () => {
        await new Promise(r => requestAnimationFrame(() => { window.__ZTA_RECORD = true; requestAnimationFrame(() => { window.__ZTA_RECORD = false; r(); }); }));
        return window.__ZTA_frames().map(f => ({ i: f.i, w: f.w, h: f.h, issues: window.__ZTA_CHECK(f.ops, f.w, f.h) }));
      });
      for (const f of frames) for (const is of f.issues) {
        const k = f.i + '|' + is.kind + '|' + is.detail;
        if (!seen.has(k)) seen.set(k, { i: f.i, kind: is.kind, detail: is.detail, scene });
      }
    };

    await capture('initial');
    /* sweep every slider together, then each on its own, then every button and
       every select option — the states a reader can actually reach */
    const nRanges = await page.evaluate(() => document.querySelectorAll('.figure input[type=range]').length);
    for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
      await page.evaluate((fr) => {
        document.querySelectorAll('.figure input[type=range]').forEach(el => {
          const lo = +el.min, hi = +el.max, st = +el.step || 1;
          el.value = String(Math.min(hi, lo + Math.round((lo + fr * (hi - lo) - lo) / st) * st));
          el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true }));
        });
      }, frac);
      await page.waitForTimeout(90);
      await capture('all sliders at ' + Math.round(frac * 100) + '%');
    }
    for (let r = 0; r < Math.min(nRanges, 40); r++) {
      for (const frac of [0, 1]) {
        await page.evaluate(([idx, fr]) => {
          const el = document.querySelectorAll('.figure input[type=range]')[idx]; if (!el) return;
          const lo = +el.min, hi = +el.max, st = +el.step || 1;
          el.value = String(Math.min(hi, lo + Math.round((lo + fr * (hi - lo) - lo) / st) * st));
          el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true }));
        }, [r, frac]);
        await page.waitForTimeout(70);
        await capture('slider ' + r + ' at ' + Math.round(frac * 100) + '%');
      }
    }
    const nSel = await page.evaluate(() => document.querySelectorAll('.figure select').length);
    for (let s = 0; s < Math.min(nSel, 30); s++) {
      const nOpt = await page.evaluate((i) => document.querySelectorAll('.figure select')[i].options.length, s);
      for (let o = 0; o < Math.min(nOpt, 12); o++) {
        const v = await page.evaluate(([i, j]) => {
          const el = document.querySelectorAll('.figure select')[i];
          el.selectedIndex = j; el.dispatchEvent(new Event('change', { bubbles: true }));
          return el.options[j].textContent.slice(0, 22);
        }, [s, o]);
        await page.waitForTimeout(70);
        await capture('select ' + JSON.stringify(v));
      }
    }
    const nBtn = await page.evaluate(() => document.querySelectorAll('.figure button').length);
    for (let b = 0; b < Math.min(nBtn, 80); b++) {
      const lab = await page.evaluate((i) => {
        const el = document.querySelectorAll('.figure button')[i]; if (!el) return '';
        const t = el.textContent.slice(0, 24); el.click(); return t;
      }, b);
      await page.waitForTimeout(70);
      await capture('after button ' + JSON.stringify(lab));
    }

    if (SHOTS) {
      const figs = await page.$$('.figure');
      for (let i = 0; i < figs.length; i++) {
        try { await figs[i].screenshot({ path: path.join(SHOT_DIR, id + '-fig' + String(i + 1).padStart(2, '0') + '.png') }); } catch (e) {}
      }
    }

    const issues = [...seen.values()];
    total += issues.length;
    if (consoleErrors.length) console.log('JSERROR ' + id + ': ' + consoleErrors.slice(0, 3).join(' | '));
    if (!issues.length) { console.log('ok   ' + id); continue; }
    bad++;
    console.log('RENDER ' + id + ' — ' + issues.length + ' issue(s)');
    const byCanvas = new Map();
    for (const is of issues) { if (!byCanvas.has(is.i)) byCanvas.set(is.i, []); byCanvas.get(is.i).push(is); }
    for (const [i, list] of [...byCanvas.entries()].sort((a, b) => a[0] - b[0])) {
      const lab = (labels.find(l => l.i === i) || {}).cap || '?';
      console.log('  canvas #' + i + '  “' + lab + '”  (' + list.length + ')');
      for (const is of list.slice(0, 14)) console.log('      [' + is.kind + '] ' + is.detail + '   — ' + is.scene);
      if (list.length > 14) console.log('      … and ' + (list.length - 14) + ' more');
    }
  }

  await browser.close(); srv.close();
  console.log(total ? '\n' + total + ' rendering issue(s) in ' + bad + ' chapter(s)' : '\nno rendering issues');
  process.exit(total ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
