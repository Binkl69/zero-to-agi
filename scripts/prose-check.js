#!/usr/bin/env node
/* Look for the wreckage an edit leaves behind.

   Rewriting one clause and leaving the rest is the commonest way this course
   has broken its own prose: a connective with nothing to connect, a word typed
   twice across a join, or — the one that actually shipped — an old sentence
   left standing next to the new one that replaced it. None of that trips the
   smoke test, because the JavaScript is still valid; a reader is the only
   thing that notices.

   These are narrow, high-signal patterns, not a grammar checker. In
   particular it cannot see a sentence fragment that reads as English on its
   own; the repeated-phrase rule is what caught the one real case of that, and
   only because the fragment repeated a phrase from the sentence before it.

   Usage: node scripts/prose-check.js */
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'prose-'));
execFileSync(process.execPath, [path.join(__dirname, 'content-inventory.js'), '--json', dir], { stdio: 'ignore' });

/* pairs that are never right in English, unlike "but because" or "and then" */
const BAD_PAIR = /\b(because|which|while|although|since|whereas|whether)\s+(<\/?[a-z][^>]*>\s*)*(and|or|but)\b/i;
const DOUBLE_OK = new Set(['that', 'had', 'is', 'no', 'so', 'very', 'did', 'you', 'the']);
/* words too common to make a repeated phrase interesting */
const BORING = new Set(['the', 'a', 'an', 'of', 'to', 'in', 'on', 'is', 'it', 'and', 'or', 'for', 'that',
  'this', 'with', 'as', 'at', 'by', 'from', 'you', 'your', 'its', 'not', 'but', 'are', 'was', 'be', 'one']);

function repeatedPhrase(text) {
  const w = text.toLowerCase().replace(/<[^>]+>/g, ' ').replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean);
  const seen = new Map();
  for (let i = 0; i + 3 <= w.length; i++) {
    const g = w.slice(i, i + 3);
    if (g.some(x => BORING.has(x))) continue;   /* all three words distinctive, or deliberate rhetorical repetition drowns the signal */
    const k = g.join(' ');
    if (seen.has(k) && i - seen.get(k) >= 4) return k;
    if (!seen.has(k)) seen.set(k, i);
  }
  return null;
}

const RULES = [
  { name: 'connective with nothing to connect', test: (t) => (t.match(BAD_PAIR) || [null])[0] },
  { name: 'doubled word', test: (t) => { const m = t.match(/\b([a-z]{3,})\s+\1\b/i); return m && !DOUBLE_OK.has(m[1].toLowerCase()) ? m[0] : null; } },
  { name: 'space before punctuation', test: (t) => (t.match(/[a-z] +[.,;](\s|$)/i) || [null])[0] },
  { name: 'doubled space', test: (t) => (t.match(/[a-z]  +[a-z]/i) || [null])[0] },
  { name: 'phrase said twice in one block', test: repeatedPhrase },
];

let hits = 0;
for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.json') && !f.startsWith('_'))) {
  const e = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
  const blocks = [
    ...e.prose.map(p => ({ where: p.kind, text: p.text })),
    ...e.captions.map(c => ({ where: 'caption', text: typeof c === 'string' ? c : c.text })),
    ...e.quizzes.flatMap(q => q.questions.flatMap(qq =>
      [{ where: 'quiz q', text: qq.q }, { where: 'quiz explain', text: qq.explain }])),
  ];
  for (const b of blocks) {
    if (!b.text) continue;
    for (const r of RULES) {
      const found = r.test(b.text);
      if (!found) continue;
      const i = b.text.toLowerCase().indexOf(String(found).split(' ')[0]);
      console.log(`${e.id}  [${r.name}]  (${b.where})  ${JSON.stringify(String(found).slice(0, 60))}`
        + `\n    …${b.text.slice(Math.max(0, i - 50), i + 130).replace(/\s+/g, ' ')}…`);
      hits++;
    }
  }
}
fs.rmSync(dir, { recursive: true, force: true });
console.log(hits ? `\n${hits} suspect passage(s)` : '\nno suspect prose');
process.exit(hits ? 1 : 0);
