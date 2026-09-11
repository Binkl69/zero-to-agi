/* Zero → AGI core runtime: chapter registry, router, and helper toolkit.
   Every chapter file calls ZTA.registerChapter({...}). The helpers in `ctx` (passed to render)
   are the only API chapters need; they handle DOM building, canvases, animation loops,
   controls, quizzes and cleanup. */
(function () {
  const ZTA = (window.ZTA = window.ZTA || {});
  ZTA.chapters = [];
  ZTA.parts = [
    { id: 'I',   title: 'Foundations',            desc: 'What "learning" means, how a network learns, and the recipe every model follows.' },
    { id: 'II',  title: 'The Model Zoo',          desc: 'The major architectures, why each was invented, and what problem each one solved.' },
    { id: 'III', title: 'How We Teach Models',    desc: 'The full pipeline that turns a pile of text and GPUs into an assistant like Claude.' },
    { id: 'IV',  title: 'Build It Yourself',      desc: 'Train models in your browser and in Python, from a single neuron to a tiny GPT.' },
    { id: 'V',   title: 'The Frontier',           desc: 'The 80-year timeline, what today\'s models still can\'t do, and the road to AGI.' },
  ];

  ZTA.registerChapter = function (ch) {
    if (!ch.id || !ch.render) throw new Error('chapter needs id and render');
    ZTA.chapters.push(ch);
    ZTA.chapters.sort((a, b) => a.num - b.num);
  };

  /* ---------- progress (localStorage) ---------- */
  const store = {
    get(k, d) { try { const v = localStorage.getItem('zta:' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('zta:' + k, JSON.stringify(v)); } catch (e) {} },
  };
  ZTA.store = store;
  ZTA.isDone = (id) => !!store.get('done', {})[id];
  ZTA.setDone = (id, v) => { const d = store.get('done', {}); if (v) d[id] = true; else delete d[id]; store.set('done', d); ZTA.onProgress && ZTA.onProgress(); };

  /* ---------- DOM builder ---------- */
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    if (attrs && typeof attrs === 'object' && !(attrs instanceof Node) && !Array.isArray(attrs)) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v == null || v === false) continue;
        if (k === 'class' || k === 'className') el.className = v;
        else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
        else if (k === 'html') el.innerHTML = v;
        else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
        else if (k === 'dataset') Object.assign(el.dataset, v);
        else el.setAttribute(k, v === true ? '' : v);
      }
    } else if (attrs != null) children.unshift(attrs);
    for (const c of children.flat(Infinity)) {
      if (c == null || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }
  ZTA.h = h;

  /* ---------- helper factory (one per chapter render, handles cleanup) ---------- */
  ZTA.makeCtx = function () {
    const cleanups = [];
    function fmt(v, o) {
      if (o.fmt) return o.fmt(v);
      const step = o.step == null ? 1 : +o.step;
      if (Number.isInteger(step) && Number.isInteger(+v)) return String(v);
      return (+v).toFixed(o.digits != null ? o.digits : 2);
    }
    const ctx = {
      h,
      /* text blocks: accept HTML strings */
      p: (html) => h('p', { html }),
      section: (title, ...kids) => h('section', {}, h('h2', {}, title), ...kids),
      sub: (title, ...kids) => h('div', {}, h('h3', {}, title), ...kids),
      ul: (items) => h('ul', {}, items.map(i => h('li', { html: i }))),
      ol: (items) => h('ol', {}, items.map(i => h('li', { html: i }))),
      callout: (kind, title, html) => h('div', { class: 'callout ' + kind }, h('div', { class: 'callout-title' }, title), typeof html === 'string' ? h('div', { html }) : html),
      code: (text, lang) => h('pre', { class: 'code', dataset: { lang: lang || '' } }, h('code', {}, text)),
      table: (headers, rows) => h('div', { class: 'table-wrap' }, h('table', { class: 'tbl' },
        h('thead', {}, h('tr', {}, headers.map(x => h('th', { html: x })))),
        h('tbody', {}, rows.map(r => h('tr', {}, r.map(c => h('td', { html: String(c) }))))))),
      cards: (items) => h('div', { class: 'grid-2' }, items.map(it => h('div', { class: 'card' }, h('h4', { html: it.title }), h('p', { html: it.body })))),
      pill: (text, color) => h('span', { class: 'pill ' + (color || '') }, text),

      /* canvas: logical size w×h, DPR-aware, scales to container width. Returns [canvasEl, ctx2d]. */
      canvas: (w, hgt) => {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const el = h('canvas', { width: Math.round(w * dpr), height: Math.round(hgt * dpr) });
        el.style.aspectRatio = w + ' / ' + hgt;
        const c = el.getContext('2d');
        c.scale(dpr, dpr);
        el.W = w; el.H = hgt;
        // map a mouse/touch event to logical coords
        el.pos = (ev) => {
          const r = el.getBoundingClientRect();
          const t = ev.touches ? ev.touches[0] : ev;
          return { x: (t.clientX - r.left) / r.width * w, y: (t.clientY - r.top) / r.height * hgt };
        };
        return [el, c];
      },
      /* figure wrapper: body (canvas or any element), optional caption, optional controls array, optional readout */
      figure: (body, caption, controls, readout) => {
        const f = h('div', { class: 'figure' });
        if (body) f.append(body.tagName === 'CANVAS' ? body : h('div', { class: 'figure-body' }, body));
        if (readout) f.append(readout);
        if (controls && controls.length) f.append(h('div', { class: 'controls' }, controls));
        if (caption) f.append(h('div', { class: 'figure-caption', html: caption }));
        return f;
      },
      readout: () => {
        const el = h('div', { class: 'readout' });
        el.set = (obj) => { el.innerHTML = ''; for (const [k, v] of Object.entries(obj)) el.append(h('span', {}, k + ': ', h('b', {}, String(v)))); };
        return el;
      },
      /* slider control; returns element with .value getter/setter */
      slider: (opts) => {
        const val = h('b', {}, fmt(opts.value, opts));
        const inp = h('input', { type: 'range', min: opts.min, max: opts.max, step: opts.step || 1, value: opts.value });
        const wrap = h('div', { class: 'control' }, h('label', {}, opts.label, val), inp);
        inp.addEventListener('input', () => { val.textContent = fmt(+inp.value, opts); opts.onChange && opts.onChange(+inp.value); });
        Object.defineProperty(wrap, 'value', { get: () => +inp.value, set: (v) => { inp.value = v; val.textContent = fmt(+v, opts); } });
        return wrap;
      },
      select: (opts) => {
        const sel = h('select', {}, opts.options.map(o => {
          const v = (o && o.value != null) ? o.value : o;
          return h('option', { value: v, selected: v === opts.value }, (o && o.label) || o);
        }));
        const wrap = h('div', { class: 'control' }, h('label', {}, opts.label), sel);
        sel.addEventListener('change', () => opts.onChange && opts.onChange(sel.value));
        Object.defineProperty(wrap, 'value', { get: () => sel.value });
        return wrap;
      },
      button: (label, onClick, cls) => h('button', { class: 'btn small ' + (cls || ''), onclick: onClick }, label),
      textarea: (opts) => {
        const ta = h('textarea', { placeholder: opts.placeholder || '' }, opts.value || '');
        const wrap = h('div', { class: 'control', style: { minWidth: '100%' } }, h('label', {}, opts.label), ta);
        ta.addEventListener('input', () => opts.onChange && opts.onChange(ta.value));
        Object.defineProperty(wrap, 'value', { get: () => ta.value, set: (v) => { ta.value = v; } });
        return wrap;
      },

      /* animation loop: fn(dt, t) — auto-stopped when the chapter unmounts. Returns controller. */
      loop: (fn) => {
        let running = true, raf = 0, last = performance.now(), t0 = last;
        const step = (now) => {
          if (!running) return;
          const dt = Math.min(0.05, (now - last) / 1000); last = now;
          try { fn(dt, (now - t0) / 1000); } catch (e) { console.error(e); running = false; return; }
          raf = requestAnimationFrame(step);
        };
        raf = requestAnimationFrame(step);
        const ctl = {
          stop() { running = false; cancelAnimationFrame(raf); },
          start() { if (!running) { running = true; last = performance.now(); raf = requestAnimationFrame(step); } },
          get running() { return running; },
        };
        cleanups.push(() => ctl.stop());
        return ctl;
      },
      interval: (fn, ms) => { const id = setInterval(fn, ms); cleanups.push(() => clearInterval(id)); return () => clearInterval(id); },
      onCleanup: (fn) => cleanups.push(fn),

      /* tabs: [{label, content: Node|()=>Node}] */
      tabs: (items) => {
        const bar = h('div', { class: 'tab-bar' }), panel = h('div', { class: 'tab-panel' });
        const btns = items.map((it, i) => h('button', { onclick: () => show(i) }, it.label));
        bar.append(...btns);
        function show(i) { btns.forEach((b, j) => b.classList.toggle('active', i === j)); panel.innerHTML = ''; const c = items[i].content; panel.append(typeof c === 'function' ? c() : c); }
        show(0);
        return h('div', { class: 'tabs' }, bar, panel);
      },

      /* decoder: a formula whose every symbol is clickable.
         parts: [{sym, name, says, points}] — sym is the symbol as written, name is what it is
         called out loud, says is what it means in plain words, points is the thing in the demo
         the reader has just used that it refers to. Plain strings are rendered as inert glue
         (brackets, equals signs) so a formula reads naturally.
         See docs/CHAPTER_CONTRACT.md: notation after the intuition, never before. */
      decoder: (parts, opts) => {
        const o = opts || {};
        const box = h('div', { class: 'decoder' });
        if (o.title) box.append(h('div', { class: 'decoder-title' }, o.title));
        const row = h('div', { class: 'decoder-formula' });
        const panel = h('div', { class: 'decoder-panel' });
        const rest = () => {
          panel.innerHTML = '';
          panel.append(h('div', { class: 'decoder-hint', html: o.hint || 'Click any symbol above. Nothing here is new — it is all naming something you have already done.' }));
        };
        const chips = [];
        parts.forEach((pt) => {
          if (typeof pt === 'string') { row.append(h('span', { class: 'decoder-glue' }, pt)); return; }
          const chip = h('button', { class: 'decoder-sym', html: pt.sym });
          chips.push(chip);
          chip.addEventListener('click', () => {
            chips.forEach(c => c.classList.toggle('active', c === chip));
            panel.innerHTML = '';
            panel.append(
              h('div', { class: 'decoder-name', html: '<b>' + pt.sym + '</b> &nbsp;is read as&nbsp; "' + pt.name + '"' }),
              h('div', { class: 'decoder-says', html: pt.says }),
              pt.points ? h('div', { class: 'decoder-points', html: '↪ ' + pt.points }) : null,
            );
          });
          row.append(chip);
        });
        rest();
        box.append(row, panel);
        if (o.plain) box.append(h('div', { class: 'decoder-plain', html: '<b>Out loud:</b> ' + o.plain }));
        return box;
      },

      /* walkthrough: a maths idea taken one step at a time, so nobody has to work out
         where to start or what order to read in. ONE step is on screen at any moment.
         steps: [{ say, note, math } | { ask, options, answer, explain }]
         Keep every `say` to one sentence. If a step needs two, it is two steps.
         See docs/CHAPTER_CONTRACT.md — this is the guided form of a maths beat. */
      walkthrough: (steps, opts) => {
        const o = opts || {};
        let i = 0;
        const box = h('div', { class: 'walk' });
        const head = h('div', { class: 'walk-head' },
          h('div', { class: 'walk-title' }, o.title || 'One step at a time'),
          h('div', { class: 'walk-count' }));
        const dots = h('div', { class: 'walk-dots' });
        const stage = h('div', { class: 'walk-stage' });
        const back = h('button', { class: 'walk-btn' }, '← Back');
        const next = h('button', { class: 'walk-btn primary' }, 'Next →');
        const nav = h('div', { class: 'walk-nav' }, back, next);

        function render() {
          const st = steps[i];
          stage.innerHTML = '';
          head.lastChild.textContent = 'step ' + (i + 1) + ' of ' + steps.length;
          dots.innerHTML = '';
          steps.forEach((_, k) => dots.append(h('span', { class: 'walk-dot' + (k === i ? ' on' : k < i ? ' seen' : '') })));

          if (st.ask) {
            stage.append(h('div', { class: 'walk-ask', html: st.ask }));
            const fb = h('div', { class: 'walk-fb' });
            const btns = st.options.map((txt, k) => {
              const b = h('button', { class: 'walk-opt', html: txt });
              b.addEventListener('click', () => {
                btns.forEach((x, j) => {
                  x.classList.toggle('right', j === st.answer);
                  x.classList.toggle('wrong', j === k && k !== st.answer);
                });
                fb.innerHTML = (k === st.answer ? '<b>Yes.</b> ' : '<b>Not quite.</b> ') + (st.explain || '');
                fb.classList.add('shown');
              });
              return b;
            });
            stage.append(h('div', { class: 'walk-opts' }, btns), fb);
          } else {
            if (st.math) stage.append(h('div', { class: 'walk-math', html: st.math }));
            stage.append(h('div', { class: 'walk-say', html: st.say }));
            if (st.note) stage.append(h('div', { class: 'walk-note', html: st.note }));
          }
          back.disabled = i === 0;
          next.textContent = i === steps.length - 1 ? 'Start again' : 'Next →';
        }
        back.addEventListener('click', () => { if (i > 0) { i--; render(); } });
        next.addEventListener('click', () => { i = (i + 1) % steps.length; render(); });
        render();
        box.append(head, dots, stage, nav);
        if (o.recap) box.append(h('div', { class: 'walk-recap', html: '<b>All of that, in one line:</b> ' + o.recap }));
        return box;
      },

      /* quiz: [{q, options:[..], answer: index, explain}] */
      quiz: (questions, title) => {
        const box = h('div', { class: 'quiz' }, h('h3', {}, title || 'Check your understanding'));
        questions.forEach((qq, qi) => {
          const opts = qq.options.map((o) => h('button', { class: 'quiz-opt', html: o }));
          const explain = h('div', { class: 'quiz-explain', style: { display: 'none' } });
          opts.forEach((b, i) => b.addEventListener('click', () => {
            opts.forEach((x, j) => { x.classList.toggle('correct', j === qq.answer); x.classList.toggle('wrong', j === i && i !== qq.answer); });
            explain.style.display = ''; explain.innerHTML = (i === qq.answer ? '✅ Correct. ' : '❌ Not quite. ') + (qq.explain || '');
          }));
          box.append(h('div', { class: 'quiz-q' }, h('div', { class: 'q' }, (qi + 1) + '. ' + qq.q), opts, explain));
        });
        return box;
      },

      /* small numeric helpers */
      rand: (a = 0, b = 1) => a + Math.random() * (b - a),
      randn: () => { let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); },
      clamp: (x, a, b) => Math.max(a, Math.min(b, x)),
      lerp: (a, b, t) => a + (b - a) * t,
      colors: { accent: '#7c9cff', green: '#38d9a9', warn: '#fbbf24', danger: '#fb7185', pink: '#f472b6', purple: '#a78bfa', orange: '#fb923c', muted: '#94a3b8', text: '#e6ebf5', bg: '#0a0e16', line: '#243044' },
      /* map value in [-1,1] to blue (negative) … red (positive) with alpha = magnitude */
      heat: (v) => { v = Math.max(-1, Math.min(1, v)); return v < 0 ? 'rgba(124,156,255,' + (-v) + ')' : 'rgba(251,113,133,' + v + ')'; },
      _cleanup: () => { cleanups.splice(0).forEach(f => { try { f(); } catch (e) {} }); },
    };
    return ctx;
  };
})();
