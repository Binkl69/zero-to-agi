/* Zero → AGI shell: navigation, routing, and everything a chapter gets for free.
   A chapter file only supplies content. The shell wraps it with: what you will be able to
   do, a section list in the sidebar that follows your scroll, a reading-progress bar,
   glossary definitions on every piece of jargon, a recap, a scored quiz that marks the
   chapter complete, and a review page that brings back the questions you missed. */
(function () {
  const ZTA = window.ZTA, h = ZTA.h, store = ZTA.store;
  const main = document.getElementById('main');
  const nav = document.getElementById('nav');
  const sidebar = document.getElementById('sidebar');
  const menuBtn = document.getElementById('menu-btn');
  let currentCtx = null;
  let toc = [];            // [{ el, title, link }] for the chapter on screen
  let pageCleanups = [];   // listeners owned by the current page
  let routeToken = 0;      // bumps on every navigation, so slow work can tell it is stale

  const pad2 = (n) => String(n).padStart(2, '0');
  const setMenu = (open) => { sidebar.classList.toggle('open', open); menuBtn.setAttribute('aria-expanded', open ? 'true' : 'false'); };
  menuBtn.addEventListener('click', () => setMenu(!sidebar.classList.contains('open')));
  main.addEventListener('click', () => setMenu(false));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { setMenu(false); hideTip(); } });

  const readBar = h('div', { id: 'read-progress', 'aria-hidden': 'true' }, h('div'));
  document.body.append(readBar);

  /* ---------- sidebar ---------- */
  function renderNav(activeId) {
    nav.innerHTML = '';
    for (const part of ZTA.parts) {
      const chs = ZTA.chapters.filter(c => c.part === part.id);
      if (!chs.length) continue;
      nav.append(h('div', { class: 'nav-part' }, 'Part ' + part.id + ' · ' + part.title));
      for (const c of chs) {
        const active = c.id === activeId;
        nav.append(h('a', { class: 'nav-item' + (active ? ' active' : '') + (ZTA.isDone(c.id) ? ' done' : ''), href: '#/ch/' + c.id, 'aria-current': active ? 'page' : null },
          h('span', { class: 'nav-num' }, pad2(c.num)), h('span', {}, c.title)));
        if (active && toc.length) {
          const list = h('div', { class: 'nav-toc' });
          for (const t of toc) {
            t.link = h('button', { class: 'nav-toc-item', type: 'button' }, t.title);
            t.link.addEventListener('click', () => { setMenu(false); t.el.scrollIntoView({ behavior: 'smooth', block: 'start' }); });
            list.append(t.link);
          }
          nav.append(list);
        }
      }
    }
    updateProgress();
    onScroll();
  }
  function updateProgress() {
    const n = ZTA.chapters.length, d = ZTA.chapters.filter(c => ZTA.isDone(c.id)).length;
    const pct = n ? Math.round(100 * d / n) : 0;
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-label').textContent = pct + '%';
  }
  ZTA.onProgress = () => renderNav(currentId());

  /* ---------- scroll: section spy, reading bar, remembered position ---------- */
  let scrollQueued = false, lastSaved = 0;
  function onScroll() {
    const id = currentId();
    const max = document.documentElement.scrollHeight - window.innerHeight;
    readBar.firstChild.style.width = (id && max > 0 ? Math.min(100, 100 * window.scrollY / max) : 0) + '%';
    if (!id) return;
    let cur = -1;
    for (let i = 0; i < toc.length; i++) { if (toc[i].el.getBoundingClientRect().top < 140) cur = i; else break; }
    toc.forEach((t, i) => t.link && t.link.classList.toggle('on', i === cur));
    const now = Date.now();
    if (now - lastSaved > 600) { lastSaved = now; store.set('last', { id, y: Math.round(window.scrollY) }); }
  }
  window.addEventListener('scroll', () => {
    hideTip();
    if (scrollQueued) return;
    scrollQueued = true;
    requestAnimationFrame(() => { scrollQueued = false; onScroll(); });
  }, { passive: true });

  /* ---------- glossary definitions on jargon ----------
     Chapters mark a term with <em> the first time they use it. Wherever that term is in the
     glossary, it becomes something you can hover, tap or tab to for the definition, so the
     reader never has to leave the page to recover a word from six chapters ago. */
  let gmap = null;
  function glossaryMap() {
    if (gmap) return gmap;
    gmap = new Map();
    const add = (k, t) => { k = k.toLowerCase().trim(); if (k.length > 1 && !gmap.has(k)) gmap.set(k, t); };
    for (const t of ZTA.glossary || []) {
      add(t.term, t);
      const m = t.term.match(/^(.*?)\s*\((.*)\)\s*$/);
      const base = m ? m[1] : t.term;
      if (m) { add(m[1], t); if (!/^in /.test(m[2])) add(m[2], t); }
      if (base.includes(' / ')) base.split(' / ').forEach(x => add(x, t));
    }
    return gmap;
  }
  function lookupTerm(text) {
    const g = glossaryMap();
    const k = text.toLowerCase().replace(/[“”"]/g, '').replace(/\s+/g, ' ').trim();
    const tries = [k, k.replace(/-/g, ' '), k.replace(/ /g, '-'), k.replace(/s$/, ''), k.replace(/es$/, ''), k.replace(/ies$/, 'y'), k.replace(/isation/, 'ization'), k.replace(/ization/, 'isation'), k.replace(/iser$/, 'izer'), k.replace(/izer$/, 'iser'), k + ' sampling', k + ' learning', k.replace(/ (loss|projection)$/, ''), k.replace(/^max /, ''), k.replace(/ly$/, '')];
    for (const t of tries) if (g.has(t)) return g.get(t);
    return null;
  }
  const tip = h('div', { class: 'gtip', role: 'tooltip' });
  document.body.append(tip);
  let tipFor = null;
  function hideTip() { if (tipFor) { tip.classList.remove('shown'); tipFor = null; } }
  function showTip(el, term) {
    tipFor = el;
    tip.innerHTML = '';
    tip.append(h('b', {}, term.term), h('span', { html: term.def }));
    tip.classList.add('shown');
    const r = el.getBoundingClientRect(), tw = tip.offsetWidth, th = tip.offsetHeight;
    let x = r.left + r.width / 2 - tw / 2;
    x = Math.max(10, Math.min(window.innerWidth - tw - 10, x));
    let y = r.top - th - 8;
    if (y < 8) y = r.bottom + 8;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function linkGlossary(root) {
    for (const em of root.querySelectorAll('em')) {
      if (em.closest('.quiz, a, button, .gterm')) continue;
      const term = lookupTerm(em.textContent);
      if (!term) continue;
      em.classList.add('gterm');
      em.setAttribute('tabindex', '0');
      em.setAttribute('role', 'button');
      em.setAttribute('aria-label', em.textContent + ': ' + term.def.replace(/<[^>]+>/g, ''));
      em.addEventListener('mouseenter', () => showTip(em, term));
      em.addEventListener('mouseleave', hideTip);
      em.addEventListener('focus', () => showTip(em, term));
      em.addEventListener('blur', hideTip);
      em.addEventListener('click', (e) => { e.stopPropagation(); if (tipFor === em) hideTip(); else showTip(em, term); });
    }
  }
  document.addEventListener('click', hideTip);

  /* ---------- toast ---------- */
  function toast(msg) {
    const t = h('div', { class: 'toast', role: 'status' }, msg);
    document.body.append(t);
    setTimeout(() => t.classList.add('gone'), 2600);
    setTimeout(() => t.remove(), 3200);
  }

  /* ---------- pages ---------- */
  function totalMinutes() { return ZTA.chapters.reduce((s, c) => s + (c.minutes || 0), 0); }
  function nextChapter() { return ZTA.chapters.find(c => !ZTA.isDone(c.id)) || null; }
  function resumeTarget() {
    const last = store.get('last', null);
    const ch = last && ZTA.chapters.find(c => c.id === last.id);
    return ch && !ZTA.isDone(ch.id) ? { ch, y: last.y || 0 } : null;
  }

  function home() {
    const wrap = h('div', { class: 'chapter fade-in' });
    const doneN = ZTA.chapters.filter(c => ZTA.isDone(c.id)).length;
    const res = resumeTarget(), nxt = nextChapter();
    let cta;
    if (res) {
      cta = h('a', { class: 'btn primary', href: '#/ch/' + res.ch.id }, 'Continue chapter ' + res.ch.num + ' →');
      cta.addEventListener('click', () => { try { sessionStorage.setItem('zta:resume', res.ch.id); } catch (e) {} });
    } else if (doneN && nxt) cta = h('a', { class: 'btn primary', href: '#/ch/' + nxt.id }, 'Next up: chapter ' + nxt.num + ' →');
    else if (doneN) cta = h('a', { class: 'btn primary', href: '#/review' }, 'All chapters done. Review →');
    else cta = h('a', { class: 'btn primary', href: '#/ch/' + (ZTA.chapters[0] || {}).id }, 'Start at chapter 1 →');
    const mins = totalMinutes();
    const facts = [ZTA.chapters.length + ' chapters', mins ? 'about ' + Math.round(mins / 60) + ' hours of reading and playing' : null, 'a hundred things to poke at', '12 Python labs', 'no maths beyond school'].filter(Boolean);
    wrap.append(
      h('div', { class: 'hero' },
        h('h1', {}, 'Zero → AGI'),
        h('p', { class: 'lead' }, 'An interactive, visual course on how AI actually works: from a single artificial neuron in 1958 to the models you talk to today, how they are trained, how to build your own, and what still separates us from AGI.'),
        h('div', { class: 'btn-row' }, cta,
          h('a', { class: 'btn', href: '#/path' }, 'See the learning path'),
          doneN ? h('a', { class: 'btn', href: '#/review' }, 'Review') : null),
        h('p', { class: 'hero-facts' }, facts.join(' · ')),
      ),
      h('div', { class: 'callout key' }, h('div', { class: 'callout-title' }, 'How to use this'),
        h('div', { html: 'Every chapter opens with <b>something to do before anything to read</b>: a slider, a live-training network, a puzzle. Play with each one until the behaviour stops surprising you. That moment is the understanding. Words in <em>green</em> are jargon: hover or tap any of them for a plain definition. Each chapter ends with a five-question quiz; finishing it marks the chapter complete, and whatever you miss comes back later on the <a href="#/review">Review</a> page. Progress is saved in this browser.' })),
    );
    for (const part of ZTA.parts) {
      const chs = ZTA.chapters.filter(c => c.part === part.id);
      if (!chs.length) continue;
      const block = h('div', { class: 'part-block' }, h('h2', {}, 'Part ' + part.id + ' — ' + part.title), h('p', { class: 'part-desc' }, part.desc));
      const grid = h('div', { class: 'grid-2' });
      for (const c of chs) {
        const best = ZTA.quizBest(c.id);
        const meta = [c.minutes ? '~' + c.minutes + ' min' : null, best ? 'quiz ' + best.score + '/' + best.total : null].filter(Boolean).join(' · ');
        grid.append(h('a', { class: 'card link chapter-card' + (ZTA.isDone(c.id) ? ' done' : ''), href: '#/ch/' + c.id },
          h('div', { class: 'num' }, pad2(c.num) + (ZTA.isDone(c.id) ? ' ✓' : '')),
          h('div', {}, h('h4', {}, c.title), h('p', {}, c.tagline || ''), meta ? h('div', { class: 'card-meta' }, meta) : null)));
      }
      block.append(grid);
      wrap.append(block);
    }
    wrap.append(h('hr'), h('p', { class: 'muted', html: 'Companion Python labs live in the <code class="inline">labs/</code> folder of this repository: build autograd, a neural net, a tokenizer, attention and a tiny GPT from scratch, then fine-tune it, add retrieval, build an agent and evaluate the lot. <a href="#/ch/13-build-it-yourself">Chapter 13</a> walks you through them.' }));
    return wrap;
  }

  const MILESTONES = [
    { id: 'm1', title: 'Finish Part I (Foundations)', desc: 'You can explain, in your own words, what a loss function is and why gradient descent works. You have watched a network learn XOR.' },
    { id: 'm2', title: 'Run labs 01–03 in Python', desc: 'Perceptron, your own autograd engine, and an MLP trained with it. Read every line; change the learning rate and see what breaks.' },
    { id: 'm3', title: 'Finish Part II (The Model Zoo)', desc: 'You know why CNNs, RNNs, embeddings, transformers and diffusion each exist, and what problem each fixed.' },
    { id: 'm4', title: 'Run labs 04–06: tokenizer → attention → tiny GPT', desc: 'Train a character-level GPT on Shakespeare on your own machine and watch the samples go from noise to English-ish.' },
    { id: 'm5', title: 'Finish Part III (How We Teach Models)', desc: 'You can draw the pretraining → SFT → RLHF/RL pipeline on a whiteboard and explain what each stage changes about the model.' },
    { id: 'm6', title: 'Run labs 07–08: RL and toy RLHF', desc: 'A Q-learning agent, then a reward model trained on preferences steering a policy. Same ideas as the real thing, 10,000× smaller.' },
    { id: 'm6b', title: 'Finish Part VI (From Understanding to Practice)', desc: 'You can grow a decision tree by hand, say when a neural network is the wrong tool, find a leak in a dataset, put an honest interval on an eval score, and explain why a canary release exists.' },
    { id: 'm6c', title: 'Run labs 09–12: fine-tune, retrieve, act, evaluate', desc: 'LoRA on your own tiny GPT, retrieval over real documents, an agent loop with tools, and an evaluation harness with confidence intervals. These four are the working parts of most LLM products.' },
    { id: 'm7', title: 'Read three foundational papers', desc: '"Attention Is All You Need" (2017), "Language Models are Few-Shot Learners" (GPT-3, 2020), "Training language models to follow instructions with human feedback" (InstructGPT, 2022). Chapter 15 has a full reading list.' },
    { id: 'm8', title: 'Reproduce a result', desc: 'Train nanoGPT (Karpathy) to GPT-2-small quality on OpenWebText, or fine-tune an open model (Llama / Qwen / Gemma) on a dataset you built. Cloud GPUs cost a few dollars an hour.' },
    { id: 'm9', title: 'Pick a frontier problem and go deep', desc: 'Interpretability, long-horizon agents, continual learning, evals, alignment, efficient training. Chapter 15 lists the open problems. Ship something public: a repo, a write-up, a benchmark.' },
  ];
  function pathPage() {
    const wrap = h('div', { class: 'chapter fade-in' });
    wrap.append(h('div', { class: 'chapter-kicker' }, 'Your learning path'), h('h1', {}, 'From "I feel behind" to building the thing'),
      h('p', { class: 'chapter-tagline' }, 'A realistic sequence. Each milestone builds on the previous one. Tick them off as you go; this is saved in your browser.'));
    const done = store.get('milestones', {});
    for (const m of MILESTONES) {
      const cb = h('input', { type: 'checkbox', checked: !!done[m.id] });
      const row = h('label', { class: 'milestone' + (done[m.id] ? ' checked' : '') }, cb, h('div', {}, h('div', { class: 'm-title' }, m.title), h('div', { class: 'm-desc', html: m.desc })));
      cb.addEventListener('change', () => { const d = store.get('milestones', {}); if (cb.checked) d[m.id] = true; else delete d[m.id]; store.set('milestones', d); row.classList.toggle('checked', cb.checked); });
      wrap.append(row);
    }
    wrap.append(h('div', { class: 'callout key', style: { marginTop: '2em' } }, h('div', { class: 'callout-title' }, 'Honest expectations'),
      h('div', { html: 'Understanding how frontier models work is a matter of months of focused study; this course plus the labs gets you most of the way. <b>Training</b> a frontier model yourself is not a solo project: the compute alone runs to hundreds of millions of dollars. But a single person <i>can</i> train a GPT-2-class model for a few hundred dollars, fine-tune open models that beat GPT-3.5 at a chosen task, and do research that moves the field. Chapter 15 maps the gap in detail.' })));
    const reset = h('button', { class: 'linklike', type: 'button' }, 'Reset all progress in this browser');
    reset.addEventListener('click', () => {
      if (!window.confirm('Clear completed chapters, quiz scores, review history and milestones? This cannot be undone.')) return;
      ['done', 'milestones', 'quizBest', 'qstats', 'last'].forEach(k => { try { localStorage.removeItem('zta:' + k); } catch (e) {} });
      route();
    });
    wrap.append(h('p', { class: 'muted', style: { marginTop: '2em', fontSize: '.85rem' } }, reset));
    return wrap;
  }

  function glossaryPage() {
    const wrap = h('div', { class: 'chapter fade-in' });
    const terms = (ZTA.glossary || []).slice().sort((a, b) => a.term.localeCompare(b.term));
    const list = h('div');
    const search = h('input', { class: 'search', type: 'search', placeholder: 'Search ' + terms.length + ' terms…', 'aria-label': 'Search the glossary' });
    function draw(q) {
      list.innerHTML = '';
      q = (q || '').toLowerCase();
      for (const t of terms) {
        if (q && !(t.term.toLowerCase().includes(q) || t.def.toLowerCase().includes(q))) continue;
        list.append(h('div', { class: 'glossary-term' }, h('b', {}, t.term), ' — ', h('span', { html: t.def })));
      }
      if (!list.children.length) list.append(h('p', { class: 'muted' }, 'No matches.'));
    }
    search.addEventListener('input', () => draw(search.value));
    draw('');
    wrap.append(h('div', { class: 'chapter-kicker' }, 'Reference'), h('h1', {}, 'Glossary'), h('p', { class: 'chapter-tagline' }, 'Every piece of jargon used in the course, in plain language.'), search, list);
    return wrap;
  }

  /* ---------- review: the questions you missed, brought back ----------
     Re-reading feels like learning and mostly is not; being made to recall is what makes
     an idea stay. So the quiz questions from every chapter you have finished go into one
     pool, and each round draws eight, taking first the ones you last got wrong, then the
     ones you have never answered, then the ones you have not seen for longest. */
  const quizCache = {};
  function harvestQuiz(ch) {
    if (quizCache[ch.id]) return quizCache[ch.id];
    const out = [];
    const ctx = ZTA.makeCtx(null);
    ctx.quiz = (qs) => { out.push(...qs); return h('div'); };
    ctx.loop = () => ({ stop() {}, start() {}, running: false });
    ctx.interval = () => () => {};
    try { ch.render(h('div'), ctx); } catch (e) { console.error(e); }
    ctx._cleanup();
    return (quizCache[ch.id] = out.map(q => Object.assign({ ch }, q)));
  }
  function reviewPage(token) {
    const wrap = h('div', { class: 'chapter fade-in' });
    wrap.append(h('div', { class: 'chapter-kicker' }, 'Review'), h('h1', {}, 'Make it stick'),
      h('p', { class: 'chapter-tagline' }, 'Eight questions drawn from the chapters you have finished, weighted toward the ones you got wrong. Five minutes, every few days, does more than rereading a chapter.'));
    const stage = h('div');
    wrap.append(stage);
    const pool = ZTA.chapters.filter(c => ZTA.isDone(c.id) || ZTA.quizBest(c.id));
    if (!pool.length) {
      const first = nextChapter() || ZTA.chapters[0];
      stage.append(h('div', { class: 'callout key' }, h('div', { class: 'callout-title' }, 'Nothing to review yet'),
        h('div', { html: 'Finish a chapter and its quiz, and its questions join the pool here. <a href="#/ch/' + first.id + '">Go to chapter ' + first.num + ' →</a>' })));
      return wrap;
    }
    stage.append(h('p', { class: 'muted' }, 'Gathering questions…'));
    (async () => {
      let all = [];
      for (const ch of pool) {
        await new Promise(r => setTimeout(r, 0));
        if (token !== routeToken) return;
        all = all.concat(harvestQuiz(ch));
      }
      round(all);
    })();

    function round(all) {
      const stats = store.get('qstats', {}), now = Date.now();
      const scored = all.map(q => {
        const s = stats[ZTA.qkey(q.q)];
        let p;
        if (!s) p = 2;
        else if (!s.last) p = 3;
        else p = Math.min(1, (now - (s.t || 0)) / (3 * 864e5));
        return { q, p: p + Math.random() * 0.6 };
      }).sort((a, b) => b.p - a.p);
      const qs = scored.slice(0, Math.min(8, scored.length)).map(x => x.q);
      const due = all.filter(q => { const s = stats[ZTA.qkey(q.q)]; return s && !s.last; }).length;
      let i = 0, right = 0;
      const missed = [];
      function show() {
        stage.innerHTML = '';
        if (i >= qs.length) return summary();
        const q = qs[i];
        const order = ZTA.optionOrder(q.options);
        const explain = h('div', { class: 'quiz-explain', style: { display: 'none' } });
        const next = h('button', { class: 'btn primary', style: { display: 'none' } }, i === qs.length - 1 ? 'See how you did →' : 'Next question →');
        next.addEventListener('click', () => { i++; show(); });
        let answered = false;
        const opts = order.map((oi) => h('button', { class: 'quiz-opt', html: q.options[oi] }));
        opts.forEach((b, k) => b.addEventListener('click', () => {
          const ok = order[k] === q.answer;
          opts.forEach((x, j) => { x.classList.toggle('correct', order[j] === q.answer); x.classList.toggle('wrong', j === k && !ok); });
          explain.style.display = ''; explain.innerHTML = (ok ? '✅ Correct. ' : '❌ Not quite. ') + (q.explain || '');
          next.style.display = '';
          if (!answered) { answered = true; ZTA.recordAnswer(q.q, ok); if (ok) right++; else missed.push(q); }
        }));
        stage.append(h('div', { class: 'quiz review-card' },
          h('div', { class: 'review-meta' }, 'Question ' + (i + 1) + ' of ' + qs.length + ' · from chapter ' + q.ch.num + ', ' + q.ch.title),
          h('div', { class: 'quiz-q' }, h('div', { class: 'q' }, q.q), opts, explain),
          h('div', { class: 'btn-row' }, next)));
      }
      function summary() {
        const box = h('div', { class: 'quiz review-card' }, h('h3', {}, right + ' of ' + qs.length));
        if (missed.length) {
          box.append(h('p', {}, 'These will come back first next time. The chapter each one came from:'));
          const seen = new Set();
          box.append(h('ul', {}, missed.filter(q => !seen.has(q.ch.id) && seen.add(q.ch.id)).map(q => h('li', {}, h('a', { href: '#/ch/' + q.ch.id }, 'Chapter ' + q.ch.num + ': ' + q.ch.title)))));
        } else box.append(h('p', {}, 'Clean round. Come back in a few days; by then the easy ones will have had time to fade, which is exactly when recalling them does the most good.'));
        const again = h('button', { class: 'btn primary' }, 'Another round');
        again.addEventListener('click', () => round(all));
        box.append(h('div', { class: 'btn-row' }, again, h('a', { class: 'btn', href: '#/' }, 'Home')));
        stage.append(box);
      }
      stage.innerHTML = '';
      const intro = h('p', { class: 'muted' }, all.length + ' questions in your pool from ' + pool.length + ' chapter' + (pool.length === 1 ? '' : 's') + (due ? ', ' + due + ' of them missed last time' : '') + '.');
      show();
      stage.prepend(intro);
    }
    return wrap;
  }

  /* ---------- a chapter ---------- */
  function chapterPage(ch) {
    const idx = ZTA.chapters.indexOf(ch);
    const prev = ZTA.chapters[idx - 1], next = ZTA.chapters[idx + 1];
    const part = ZTA.parts.find(p => p.id === ch.part) || {};
    const wrap = h('div', { class: 'chapter fade-in' });
    wrap.append(h('div', { class: 'chapter-kicker' }, 'Part ' + ch.part + ' · ' + part.title + ' · Chapter ' + ch.num + (ch.minutes ? ' · about ' + ch.minutes + ' min' : '')), h('h1', {}, ch.title));
    if (ch.tagline) wrap.append(h('p', { class: 'chapter-tagline' }, ch.tagline));
    if (ch.objectives && ch.objectives.length) {
      wrap.append(h('details', { class: 'objectives' },
        h('summary', {}, 'What you will be able to do after this chapter'),
        h('ul', {}, ch.objectives.map(o => h('li', { html: o })))));
    }
    try {
      if (window.matchMedia('(max-width: 600px) and (orientation: portrait)').matches && !store.get('rotateHint', false)) {
        const x = h('button', { class: 'linklike', type: 'button', 'aria-label': 'Dismiss' }, 'Got it');
        const hint = h('div', { class: 'rotate-hint' }, h('span', {}, 'The demos are detailed. When one feels cramped, turn your phone sideways.'), x);
        x.addEventListener('click', () => { store.set('rotateHint', true); hint.remove(); });
        wrap.append(hint);
      }
    } catch (e) {}

    const body = h('div', { class: 'chapter-body' });
    currentCtx = ZTA.makeCtx(ch.id);
    try { ch.render(body, currentCtx); } catch (e) { console.error(e); body.append(h('div', { class: 'callout warning' }, h('div', { class: 'callout-title' }, 'Chapter failed to render'), h('pre', { class: 'code' }, String(e.stack || e)))); }
    wrap.append(body);

    toc = [];
    for (const h2 of body.querySelectorAll('h2')) toc.push({ el: h2, title: h2.textContent });
    linkGlossary(body);

    const doneBtn = h('button', { class: 'btn' });
    const syncDone = () => { const v = ZTA.isDone(ch.id); doneBtn.className = 'btn' + (v ? ' done' : ' primary'); doneBtn.textContent = v ? '✓ Completed' : 'Mark complete'; };
    syncDone();
    doneBtn.addEventListener('click', () => { ZTA.setDone(ch.id, !ZTA.isDone(ch.id)); syncDone(); });
    ZTA.onQuizDone = (id) => {
      if (id !== ch.id || ZTA.isDone(ch.id)) return;
      ZTA.setDone(ch.id, true); syncDone();
      toast('Chapter ' + ch.num + ' marked complete');
    };

    if (ch.takeaways && ch.takeaways.length) {
      const quiz = body.querySelector('.quiz');
      const box = h('div', { class: 'takeaways' }, h('div', { class: 'takeaways-title' }, 'The chapter in ' + ch.takeaways.length + ' lines'),
        h('ul', {}, ch.takeaways.map(t => h('li', { html: t }))));
      /* the recap goes just ahead of the quiz when the quiz is the chapter's own top-level
         block, so it is read as the lead-in to being tested rather than as an appendix */
      if (quiz && quiz.parentNode === body) body.insertBefore(box, quiz); else body.append(box);
      linkGlossary(box);
    }

    wrap.append(h('div', { class: 'chapter-nav' },
      prev ? h('a', { href: '#/ch/' + prev.id }, '← ' + prev.title) : h('a', { href: '#/' }, '← Home'),
      doneBtn,
      next ? h('a', { href: '#/ch/' + next.id }, next.title + ' →') : h('a', { href: '#/path' }, 'Learning path →')));
    return wrap;
  }

  /* ---------- router ---------- */
  function currentId() { const m = location.hash.match(/^#\/ch\/([\w-]+)/); return m ? m[1] : null; }
  function route() {
    routeToken++;
    if (currentCtx) { currentCtx._cleanup(); currentCtx = null; }
    pageCleanups.splice(0).forEach(f => { try { f(); } catch (e) {} });
    ZTA.onQuizDone = null;
    toc = [];
    hideTip();
    setMenu(false);
    main.innerHTML = '';
    const hash = location.hash || '#/';
    let page, restoreY = 0;
    if (hash.startsWith('#/ch/')) {
      const ch = ZTA.chapters.find(c => c.id === currentId());
      page = ch ? chapterPage(ch) : h('div', { class: 'chapter' }, h('h1', {}, 'Chapter not found'), h('a', { href: '#/' }, 'Home'));
      try {
        if (ch && sessionStorage.getItem('zta:resume') === ch.id) { const last = store.get('last', null); if (last && last.id === ch.id) restoreY = last.y || 0; }
        sessionStorage.removeItem('zta:resume');
      } catch (e) {}
    } else if (hash.startsWith('#/path')) page = pathPage();
    else if (hash.startsWith('#/glossary')) page = glossaryPage();
    else if (hash.startsWith('#/review')) page = reviewPage(routeToken);
    else page = home();
    main.append(page);
    renderNav(currentId());
    /* with twenty chapters the active one can sit below the sidebar's fold */
    const activeNav = nav.querySelector('.nav-item.active');
    if (activeNav && activeNav.scrollIntoView) activeNav.scrollIntoView({ block: 'nearest' });
    window.scrollTo(0, restoreY);
    document.title = (page.querySelector('h1') ? page.querySelector('h1').textContent + ' · ' : '') + 'Zero → AGI';
  }
  window.addEventListener('hashchange', route);
  route();
})();
