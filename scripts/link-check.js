#!/usr/bin/env node
/* Check every external link the course sends a reader to.

   A dead link in a "go deeper" list is a small thing that makes a course feel
   abandoned, and a link that silently moved is worse — the reader lands
   somewhere that does not say what we claimed it says. This reports status,
   redirects and the page title, so both can be caught.

   Usage: node scripts/link-check.js [--json <file>] */
const { execFile } = require('child_process');
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');
const CONC = 6;

const files = []
  .concat(fs.readdirSync(path.join(ROOT, 'app/chapters')).map(f => path.join(ROOT, 'app/chapters', f)))
  .concat(fs.readdirSync(path.join(ROOT, 'app/js')).map(f => path.join(ROOT, 'app/js', f)))
  .concat([path.join(ROOT, 'index.html')])
  .filter(f => /\.(js|html)$/.test(f));

const seen = new Map();
for (const f of files) {
  const src = fs.readFileSync(f, 'utf8');
  const re = /<a\s+href="(https?:\/\/[^"]+)"[^>]*>([\s\S]*?)<\/a>/g;
  let m;
  while ((m = re.exec(src))) {
    const url = m[1], text = m[2].replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
    if (!seen.has(url)) seen.set(url, { url, texts: [], where: [] });
    const e = seen.get(url);
    if (!e.texts.includes(text)) e.texts.push(text);
    const w = path.basename(f);
    if (!e.where.includes(w)) e.where.push(w);
  }
}
const links = [...seen.values()];

function probe(url) {
  return new Promise((res) => {
    /* -L follows redirects; the write-out reports the code and the final URL so
       a silent move is visible. A HEAD is enough for status; the body is only
       fetched to read <title>. */
    execFile('curl', ['-sS', '-L', '--max-time', '25', '--compressed',
      '-A', 'Mozilla/5.0 (compatible; zero-to-agi link check)',
      '-o', '-', '-w', '\\n@@@%{http_code}@@@%{url_effective}@@@', url],
    { maxBuffer: 8 * 1024 * 1024 }, (err, stdout) => {
      const body = String(stdout || '');
      const m = body.match(/\n@@@(\d+)@@@([^@]*)@@@$/);
      const code = m ? +m[1] : 0;
      const finalUrl = m ? m[2] : '';
      const t = body.match(/<title[^>]*>([\s\S]{0,300}?)<\/title>/i);
      const title = t ? t[1].replace(/\s+/g, ' ').trim() : '';
      res({ code, finalUrl, title, err: err ? String(err.message).split('\n')[0] : null });
    });
  });
}

(async () => {
  const out = [];
  let i = 0;
  await Promise.all(Array.from({ length: CONC }, async () => {
    while (i < links.length) {
      const L = links[i++];
      const r = await probe(L.url);
      out.push(Object.assign({}, L, r));
      process.stderr.write('.');
    }
  }));
  process.stderr.write('\n');
  out.sort((a, b) => a.url.localeCompare(b.url));
  let bad = 0, moved = 0;
  for (const r of out) {
    const movedTo = r.finalUrl && r.finalUrl.replace(/\/$/, '') !== r.url.replace(/\/$/, '') ? r.finalUrl : null;
    if (r.code < 200 || r.code >= 400) {
      bad++;
      console.log('DEAD  ' + r.code + '  ' + r.url + '\n        in ' + r.where.join(', ') + '  as ' + JSON.stringify(r.texts[0] || '') + (r.err ? '\n        ' + r.err : ''));
    } else if (movedTo) {
      moved++;
      console.log('MOVED ' + r.code + '  ' + r.url + '\n        -> ' + movedTo + '\n        title: ' + r.title.slice(0, 90));
    }
  }
  console.log('\n' + out.length + ' links: ' + (out.length - bad - moved) + ' fine, ' + moved + ' redirected, ' + bad + ' dead');
  const jat = process.argv.indexOf('--json');
  if (jat >= 0) { fs.writeFileSync(process.argv[jat + 1], JSON.stringify(out, null, 1)); console.log('wrote ' + process.argv[jat + 1]); }
  process.exit(bad ? 1 : 0);
})();
