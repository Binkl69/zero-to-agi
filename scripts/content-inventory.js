#!/usr/bin/env node
/* Pull every checkable claim out of the chapters into one machine-readable pile.

   The rendering checkers ask whether the picture is drawn correctly. This asks
   what the course actually says, so the saying can be checked: the prose, the
   figure captions, the quiz questions with their marked answers, every external
   link, and every number that appears in prose (a number in prose is a promise
   that some demo, paper or measurement backs it up).

   Usage: node scripts/content-inventory.js [--json <dir>] [--check] [chapter-id ...] */
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const makeEnv = require('./lib/fakedom.js');

const ROOT = path.join(__dirname, '..');
const argv = process.argv.slice(2);
const CHECK = argv.includes('--check');
const jsonAt = argv.indexOf('--json');
const OUT = jsonAt >= 0 ? argv[jsonAt + 1] : null;
const only = argv.filter((a, i) => !a.startsWith('--') && i !== jsonAt + 1);

const env = makeEnv();
const { window, Element, walk, rafQueue, timers } = env;
const context = vm.createContext(window);
const load = (f) => vm.runInContext(fs.readFileSync(f, 'utf8'), context, { filename: f });
load(path.join(ROOT, 'app/js/core.js'));

const files = fs.readdirSync(path.join(ROOT, 'app/chapters')).filter(f => f.endsWith('.js')).sort();
const report = [];
let problems = 0;

for (const f of files) {
  const id = f.replace(/\.js$/, '');
  if (only.length && !only.some(o => id.includes(o))) continue;
  const src = fs.readFileSync(path.join(ROOT, 'app/chapters', f), 'utf8');
  rafQueue.length = 0; timers.length = 0;
  load(path.join(ROOT, 'app/chapters', f));
  const ch = window.ZTA.chapters.find(c => c.id === id);
  const ctx = window.ZTA.makeCtx();

  /* capture the quiz payload as the chapter hands it over, rather than trying
     to read the answers back out of the rendered buttons */
  const quizzes = [];
  const realQuiz = ctx.quiz;
  ctx.quiz = (questions, title) => { quizzes.push({ title: title || null, questions }); return realQuiz(questions, title); };
  const captions = [];
  const realFigure = ctx.figure;
  ctx.figure = (body, caption, controls, readout) => { if (caption) captions.push(String(caption)); return realFigure(body, caption, controls, readout); };

  const root = new Element('div');
  ch.render(root, ctx);
  for (let i = 0; i < 4; i++) { const q = rafQueue.splice(0); for (const fn of q) { try { fn(Date.now() + i * 16); } catch (e) {} } }
  try { ctx._cleanup(); } catch (e) {}
  rafQueue.length = 0; timers.length = 0;

  /* prose: every paragraph, callout, card and list item the reader reads */
  const prose = [];
  const grab = (el, kind) => {
    const t = el.textContent.replace(/\s+/g, ' ').trim();
    if (t) prose.push({ kind, text: t });
  };
  walk(root, el => {
    if (el.tagName === 'P') grab(el, 'paragraph');
    else if (el.classList.contains('callout')) grab(el, 'callout');
    else if (el.tagName === 'LI') grab(el, 'list-item');
  });

  const links = [];
  const LINK = /<a\s+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g;
  let m;
  while ((m = LINK.exec(src))) {
    if (!/^https?:/.test(m[1])) continue;
    links.push({ url: m[1], text: m[2].replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim() });
  }

  /* numbers in prose, with the sentence they sit in — this is the checklist a
     fact-checker works from: each one is a claim something has to back */
  const numbers = [];
  const NUM = /(?:^|[\s(>—–-])(\$?£?€?\d[\d,]*(?:\.\d+)?\s*(?:%|×|x\b|B\b|M\b|K\b|bn\b|billion|million|trillion|thousand|GB|TB|MB|FLOPs?|tokens?|params?|parameters|layers?|years?|GPUs?|hours?|days?)?)/g;
  for (const p of prose.concat(captions.map(c => ({ kind: 'caption', text: String(c).replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim() })))) {
    const sentences = p.text.split(/(?<=[.!?])\s+/);
    for (const s of sentences) {
      let mm; NUM.lastIndex = 0;
      while ((mm = NUM.exec(s))) {
        const v = mm[1].trim();
        if (/^\d{1,2}$/.test(v) && !/%/.test(v)) continue;   /* bare small integers are usually prose, not claims */
        numbers.push({ value: v, sentence: s.trim(), where: p.kind });
      }
    }
  }

  /* Controls the reader can actually touch, and the controls the prose tells them
     to touch. A "try this" step that names a button nobody built is a dead end
     the reader blames on themselves. */
  const controls = [];
  walk(root, el => {
    if (el.tagName === 'BUTTON') controls.push(el.textContent);
    else if (el.tagName === 'OPTION') controls.push(el.textContent);
    else if (el.classList.contains('control')) {
      const lab = el.querySelector('label');
      if (lab) controls.push(lab.textContent);
    }
  });
  const norm = (t) => String(t).toLowerCase().replace(/[\u{1F300}-\u{1FAFF}\u{2190}-\u{27BF}]/gu, ' ')
    .replace(/[^a-z0-9+\-. ]/g, ' ').replace(/\s+/g, ' ').trim();
  const controlSet = controls.map(norm).filter(Boolean);
  const IMPERATIVE = /\b(press|click|hit|tap|drag|set|choose|select|switch to|turn on|toggle|tick|slide)\b[^<]{0,24}?<b>([^<]{2,44})<\/b>/gi;
  const dangling = [];
  /* innerHTML only holds the node's own markup, and a callout keeps its body in
     a child, so gather the whole subtree's markup before matching. */
  const subtreeHtml = (el) => {
    let out = el.innerHTML || '';
    walk(el, c => { out += ' ' + (c.innerHTML || ''); });
    return out;
  };
  walk(root, el => {
    if (!el.classList.contains('callout')) return;
    const html = subtreeHtml(el);
    let mm; IMPERATIVE.lastIndex = 0;
    while ((mm = IMPERATIVE.exec(html))) {
      const phrase = norm(mm[2]);
      if (!phrase || phrase.split(' ').length > 6) continue;
      if (!/[a-z]/.test(phrase)) continue;                      /* "drag to 0.62" names a value, not a control */
      /* a control label must be substantial before "the phrase contains it"
         counts — otherwise a button labelled "b" matches every phrase with a b */
      if (controlSet.some(c => c === phrase || c.includes(phrase) || (c.length >= 5 && phrase.includes(c)))) continue;
      /* the prose rarely quotes a label word for word — "switch to card fraud"
         should find the option "Is this card payment fraud?" — so a control whose
         label contains every significant word of the phrase counts as a match */
      const words = phrase.split(' ').filter(w => w.length > 2 && !['the', 'and', 'for', 'with', 'this', 'that'].includes(w));
      if (words.length && controlSet.some(c => words.every(w => c.includes(w)))) continue;
      dangling.push({ phrase: mm[2].trim(), verb: mm[1], context: html.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').slice(0, 110) });
    }
  });

  const entry = {
    id, title: ch.title, part: ch.part, num: ch.num,
    counts: { prose: prose.length, captions: captions.length, links: links.length,
      quizQuestions: quizzes.reduce((n, q) => n + q.questions.length, 0), numericClaims: numbers.length,
      controls: controls.length, danglingControlRefs: dangling.length },
    prose, captions, links, quizzes, numbers, controls, dangling,
  };
  report.push(entry);

  if (CHECK) {
    for (const d of dangling) {
      problems++;
      console.log('CONTROL ' + id + ': the prose says ' + JSON.stringify(d.verb + ' ' + d.phrase)
        + ' but no control is labelled that\n         — ' + d.context);
    }
    /* structural checks on the quizzes — cheap, deterministic, and a wrong
       answer index silently teaches the reader the wrong thing */
    for (const qz of quizzes) {
      qz.questions.forEach((q, i) => {
        const at = id + ' quiz Q' + (i + 1);
        const say = (msg) => { problems++; console.log('QUIZ ' + at + ': ' + msg + '  — ' + String(q.q).slice(0, 70)); };
        if (!q.q || !String(q.q).trim()) say('empty question');
        if (!Array.isArray(q.options) || q.options.length < 2) say('fewer than two options');
        else {
          if (!(Number.isInteger(q.answer) && q.answer >= 0 && q.answer < q.options.length)) say('answer index ' + q.answer + ' is outside 0..' + (q.options.length - 1));
          const seen = new Set();
          for (const o of q.options) {
            const k = String(o).toLowerCase().replace(/\s+/g, ' ').trim();
            if (seen.has(k)) say('duplicate option ' + JSON.stringify(String(o).slice(0, 40)));
            seen.add(k);
            if (!k) say('blank option');
          }
        }
        if (!q.explain || String(q.explain).trim().length < 20) say('missing or trivial explanation');
      });
    }
  }
}

if (OUT) {
  fs.mkdirSync(OUT, { recursive: true });
  for (const e of report) fs.writeFileSync(path.join(OUT, e.id + '.json'), JSON.stringify(e, null, 1));
  fs.writeFileSync(path.join(OUT, '_all-links.json'), JSON.stringify(
    report.flatMap(e => e.links.map(l => ({ chapter: e.id, ...l }))), null, 1));
  console.log('wrote ' + report.length + ' chapter inventories to ' + OUT);
}
console.log('\nchapter                       prose  caps  links  quiz  numeric-claims');
for (const e of report) {
  const c = e.counts;
  console.log(e.id.padEnd(28) + String(c.prose).padStart(6) + String(c.captions).padStart(6)
    + String(c.links).padStart(7) + String(c.quizQuestions).padStart(6) + String(c.numericClaims).padStart(16));
}
const tot = report.reduce((a, e) => { for (const k in e.counts) a[k] = (a[k] || 0) + e.counts[k]; return a; }, {});
console.log('TOTAL'.padEnd(28) + String(tot.prose).padStart(6) + String(tot.captions).padStart(6)
  + String(tot.links).padStart(7) + String(tot.quizQuestions).padStart(6) + String(tot.numericClaims).padStart(16));
if (CHECK) console.log(problems ? '\n' + problems + ' structural quiz problem(s)' : '\nquizzes are structurally sound');
process.exit(problems ? 1 : 0);
