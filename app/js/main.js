/* Zero → AGI shell: navigation, routing, home / learning-path / glossary pages. */
(function () {
  const ZTA = window.ZTA, h = ZTA.h;
  const main = document.getElementById('main');
  const nav = document.getElementById('nav');
  const sidebar = document.getElementById('sidebar');
  let currentCtx = null;

  document.getElementById('menu-btn').addEventListener('click', () => sidebar.classList.toggle('open'));
  main.addEventListener('click', () => sidebar.classList.remove('open'));

  /* ---------- sidebar ---------- */
  function renderNav(activeId) {
    nav.innerHTML = '';
    for (const part of ZTA.parts) {
      const chs = ZTA.chapters.filter(c => c.part === part.id);
      if (!chs.length) continue;
      nav.append(h('div', { class: 'nav-part' }, 'Part ' + part.id + ' · ' + part.title));
      for (const c of chs) {
        nav.append(h('a', { class: 'nav-item' + (c.id === activeId ? ' active' : '') + (ZTA.isDone(c.id) ? ' done' : ''), href: '#/ch/' + c.id },
          h('span', { class: 'nav-num' }, String(c.num).padStart(2, '0')), h('span', {}, c.title)));
      }
    }
    updateProgress();
  }
  function updateProgress() {
    const n = ZTA.chapters.length, d = ZTA.chapters.filter(c => ZTA.isDone(c.id)).length;
    const pct = n ? Math.round(100 * d / n) : 0;
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-label').textContent = pct + '%';
  }
  ZTA.onProgress = () => renderNav(currentId());

  /* ---------- pages ---------- */
  function home() {
    const wrap = h('div', { class: 'chapter fade-in' });
    wrap.append(
      h('div', { class: 'hero' },
        h('h1', {}, 'Zero → AGI'),
        h('p', { class: 'lead' }, 'An interactive, visual course on how AI actually works: from a single artificial neuron in 1958 to the models you talk to today, how they are trained, how to build your own, and what still separates us from AGI.'),
        h('div', { class: 'btn-row' },
          h('a', { class: 'btn primary', href: '#/ch/' + (ZTA.chapters[0] || {}).id }, 'Start at chapter 1 →'),
          h('a', { class: 'btn', href: '#/path' }, 'See the learning path'),
        ),
      ),
      ZTA.h('div', { class: 'callout key' }, h('div', { class: 'callout-title' }, 'How to use this'),
        h('div', { html: 'Every chapter mixes short explanations with <b>things you can poke at</b>: sliders, live-training networks, drawing canvases, games. Play with each one until the behaviour stops surprising you. That moment is the understanding. Each chapter ends with a quick quiz and a <em>Mark complete</em> button; progress is saved in your browser.' })),
    );
    for (const part of ZTA.parts) {
      const chs = ZTA.chapters.filter(c => c.part === part.id);
      if (!chs.length) continue;
      const block = h('div', { class: 'part-block' }, h('h2', {}, 'Part ' + part.id + ' — ' + part.title), h('p', { class: 'part-desc' }, part.desc));
      const grid = h('div', { class: 'grid-2' });
      for (const c of chs) {
        grid.append(h('a', { class: 'card link chapter-card' + (ZTA.isDone(c.id) ? ' done' : ''), href: '#/ch/' + c.id },
          h('div', { class: 'num' }, String(c.num).padStart(2, '0') + (ZTA.isDone(c.id) ? ' ✓' : '')),
          h('div', {}, h('h4', {}, c.title), h('p', {}, c.tagline || ''))));
      }
      block.append(grid);
      wrap.append(block);
    }
    wrap.append(h('hr'), h('p', { class: 'muted', html: 'Companion Python labs live in the <code class="inline">labs/</code> folder of this repository: build autograd, a neural net, a tokenizer, attention and a tiny GPT from scratch. Chapter 13 walks you through them.' }));
    return wrap;
  }

  const MILESTONES = [
    { id: 'm1', title: 'Finish Part I (Foundations)', desc: 'You can explain, in your own words, what a loss function is and why gradient descent works. You have watched a network learn XOR.' },
    { id: 'm2', title: 'Run labs 01–03 in Python', desc: 'Perceptron, your own autograd engine, and an MLP trained with it. Read every line; change the learning rate and see what breaks.' },
    { id: 'm3', title: 'Finish Part II (The Model Zoo)', desc: 'You know why CNNs, RNNs, embeddings, transformers and diffusion each exist, and what problem each fixed.' },
    { id: 'm4', title: 'Run labs 04–06: tokenizer → attention → tiny GPT', desc: 'Train a character-level GPT on Shakespeare on your own machine and watch the samples go from noise to English-ish.' },
    { id: 'm5', title: 'Finish Part III (How We Teach Models)', desc: 'You can draw the pretraining → SFT → RLHF/RL pipeline on a whiteboard and explain what each stage changes about the model.' },
    { id: 'm6', title: 'Run labs 07–08: RL and toy RLHF', desc: 'A Q-learning agent, then a reward model trained on preferences steering a policy. Same ideas as the real thing, 10,000× smaller.' },
    { id: 'm7', title: 'Read three foundational papers', desc: '"Attention Is All You Need" (2017), "Language Models are Few-Shot Learners" (GPT-3, 2020), "Training language models to follow instructions with human feedback" (InstructGPT, 2022). Chapter 15 has a full reading list.' },
    { id: 'm8', title: 'Reproduce a result', desc: 'Train nanoGPT (Karpathy) to GPT-2-small quality on OpenWebText, or fine-tune an open model (Llama / Qwen / Gemma) on a dataset you built. Cloud GPUs cost a few dollars an hour.' },
    { id: 'm9', title: 'Pick a frontier problem and go deep', desc: 'Interpretability, long-horizon agents, continual learning, evals, alignment, efficient training. Chapter 15 lists the open problems. Ship something public: a repo, a write-up, a benchmark.' },
  ];
  function pathPage() {
    const wrap = h('div', { class: 'chapter fade-in' });
    wrap.append(h('div', { class: 'chapter-kicker' }, 'Your learning path'), h('h1', {}, 'From "I feel behind" to building the thing'),
      h('p', { class: 'chapter-tagline' }, 'A realistic sequence. Each milestone builds on the previous one. Tick them off as you go; this is saved in your browser.'));
    const done = ZTA.store.get('milestones', {});
    for (const m of MILESTONES) {
      const cb = h('input', { type: 'checkbox', checked: !!done[m.id] });
      const row = h('label', { class: 'milestone' + (done[m.id] ? ' checked' : '') }, cb, h('div', {}, h('div', { class: 'm-title' }, m.title), h('div', { class: 'm-desc', html: m.desc })));
      cb.addEventListener('change', () => { const d = ZTA.store.get('milestones', {}); if (cb.checked) d[m.id] = true; else delete d[m.id]; ZTA.store.set('milestones', d); row.classList.toggle('checked', cb.checked); });
      wrap.append(row);
    }
    wrap.append(h('div', { class: 'callout key', style: { marginTop: '2em' } }, h('div', { class: 'callout-title' }, 'Honest expectations'),
      h('div', { html: 'Understanding how frontier models work is a matter of months of focused study; this course plus the labs gets you most of the way. <b>Training</b> a frontier model yourself is not a solo project: the compute alone runs to hundreds of millions of dollars. But a single person <i>can</i> train a GPT-2-class model for a few hundred dollars, fine-tune open models that beat GPT-3.5 at a chosen task, and do research that moves the field. Chapter 15 maps the gap in detail.' })));
    return wrap;
  }

  function glossaryPage() {
    const wrap = h('div', { class: 'chapter fade-in' });
    const terms = (ZTA.glossary || []).slice().sort((a, b) => a.term.localeCompare(b.term));
    const list = h('div');
    const search = h('input', { class: 'search', placeholder: 'Search ' + terms.length + ' terms…' });
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

  function chapterPage(ch) {
    const idx = ZTA.chapters.indexOf(ch);
    const prev = ZTA.chapters[idx - 1], next = ZTA.chapters[idx + 1];
    const part = ZTA.parts.find(p => p.id === ch.part) || {};
    const wrap = h('div', { class: 'chapter fade-in' });
    wrap.append(h('div', { class: 'chapter-kicker' }, 'Part ' + ch.part + ' · ' + part.title + ' · Chapter ' + ch.num), h('h1', {}, ch.title));
    if (ch.tagline) wrap.append(h('p', { class: 'chapter-tagline' }, ch.tagline));
    const body = h('div');
    currentCtx = ZTA.makeCtx();
    try { ch.render(body, currentCtx); } catch (e) { console.error(e); body.append(h('div', { class: 'callout warning' }, h('div', { class: 'callout-title' }, 'Chapter failed to render'), h('pre', { class: 'code' }, String(e.stack || e)))); }
    wrap.append(body);
    const doneBtn = h('button', { class: 'btn' + (ZTA.isDone(ch.id) ? ' done' : ' primary') }, ZTA.isDone(ch.id) ? '✓ Completed' : 'Mark complete');
    doneBtn.addEventListener('click', () => { const v = !ZTA.isDone(ch.id); ZTA.setDone(ch.id, v); doneBtn.className = 'btn' + (v ? ' done' : ' primary'); doneBtn.textContent = v ? '✓ Completed' : 'Mark complete'; });
    wrap.append(h('div', { class: 'chapter-nav' },
      prev ? h('a', { href: '#/ch/' + prev.id }, '← ' + prev.title) : h('a', { href: '#/' }, '← Home'),
      doneBtn,
      next ? h('a', { href: '#/ch/' + next.id }, next.title + ' →') : h('a', { href: '#/path' }, 'Learning path →')));
    return wrap;
  }

  /* ---------- router ---------- */
  function currentId() { const m = location.hash.match(/^#\/ch\/([\w-]+)/); return m ? m[1] : null; }
  function route() {
    if (currentCtx) { currentCtx._cleanup(); currentCtx = null; }
    main.innerHTML = '';
    const hash = location.hash || '#/';
    let page;
    if (hash.startsWith('#/ch/')) {
      const ch = ZTA.chapters.find(c => c.id === currentId());
      page = ch ? chapterPage(ch) : h('div', { class: 'chapter' }, h('h1', {}, 'Chapter not found'), h('a', { href: '#/' }, 'Home'));
    } else if (hash.startsWith('#/path')) page = pathPage();
    else if (hash.startsWith('#/glossary')) page = glossaryPage();
    else page = home();
    main.append(page);
    renderNav(currentId());
    window.scrollTo(0, 0);
    document.title = (page.querySelector('h1') ? page.querySelector('h1').textContent + ' · ' : '') + 'Zero → AGI';
  }
  window.addEventListener('hashchange', route);
  route();
})();
