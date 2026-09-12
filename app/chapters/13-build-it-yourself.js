/* Zero → AGI · Chapter 13 · Build it yourself: the eight-lab ladder
   DESIGN RULE: a real language model is training in the reader's browser before any prose runs.
   Interactives, in order: the live char-level trainer; the autograd engine; tensor shape flow
   through a tiny GPT (the crash everyone hits first); the initialisation checker; and a loss
   doctor that asks the reader to diagnose five broken training runs. */
(function () {
  /* ---------------- training corpora for the live demo ---------------- */
  const PROSE = `the sun came up over the little town. the baker lit her oven and the smell of warm bread drifted down the street. a small dog sat by the door and waited. the cat on the wall watched the dog and the dog watched the bread. children ran to school with books under their arms. the old clock in the square struck eight. the fisherman pushed his boat into the cold water and the gulls followed him out to sea.
by noon the market was full of voices. one man sold red apples, another sold fish, and a girl sold flowers from a basket. the baker gave the small dog a crust and the dog wagged its tail. in the afternoon the wind picked up and the boats came home early. the fisherman had three fish and a story about a fourth. the children came home and did their sums at the kitchen table. the cat slept in a patch of sun.
when the sun went down the lamps came on one by one along the street. the baker closed her shop and walked home past the quiet square. the old clock struck nine. the dog followed her to the corner and then turned back. the town went to sleep and the stars came out over the sea.
in the morning it all began again. the sun came up, the oven was lit, the bread was warm, and the small dog waited by the door. the cat sat on the wall. the children ran to school. the fisherman pushed his boat into the water and the gulls followed him out to sea. the market filled with voices and the clock in the square struck eight, then nine, then ten. the baker smiled and gave the dog another crust. it was a good town and a good day and the small dog was very happy.
`;
  const PYCODE = `def add(a, b):
    return a + b

def mul(a, b):
    return a * b

def square(x):
    return x * x

def total(items):
    result = 0
    for item in items:
        result = result + item
    return result

def mean(items):
    if len(items) == 0:
        return 0
    return total(items) / len(items)

def largest(items):
    best = items[0]
    for item in items:
        if item > best:
            best = item
    return best

def smallest(items):
    best = items[0]
    for item in items:
        if item < best:
            best = item
    return best

def count_even(items):
    count = 0
    for item in items:
        if item % 2 == 0:
            count = count + 1
    return count

def clip(x, low, high):
    if x < low:
        return low
    if x > high:
        return high
    return x

class Counter:
    def __init__(self):
        self.value = 0

    def step(self):
        self.value = self.value + 1
        return self.value

    def reset(self):
        self.value = 0

for i in range(10):
    print(i, square(i), add(i, 1))
    if i % 2 == 0:
        print("even", i)
    else:
        print("odd", i)
`;

  /* ---------------- code snippets shown in the prose ---------------- */
  const VALUE_CODE = `class Value:
    """A number that remembers how it was made, so it can send gradients back."""
    def __init__(self, data, children=(), op=''):
        self.data, self.grad = data, 0.0
        self._backward = lambda: None      # filled in by the op that created this node
        self._prev, self._op = set(children), op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad  += out.grad          # d(out)/d(self) = 1, times what flows in
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  += other.data * out.grad   # chain rule: local derivative x upstream
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, seen = [], set()               # order nodes so parents come after children
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0                      # dL/dL = 1
        for v in reversed(topo): v._backward()`;

  const ATTN_CODE = `def attention(x, Wq, Wk, Wv):          # x: (T, d) — T tokens, each a d-vector
    T = x.shape[0]
    q, k, v = x @ Wq, x @ Wk, x @ Wv       # queries, keys, values: (T, d_head)
    scores = q @ k.T / np.sqrt(k.shape[-1])          # (T, T): how much i cares about j
    mask = np.triu(np.ones((T, T)), k=1).astype(bool)   # True = "not allowed to look"
    scores[mask] = -np.inf                  # causal: position i may not look at j > i
    w = np.exp(scores - scores.max(-1, keepdims=True))  # max is finite: j = i is never masked
    w = w / w.sum(-1, keepdims=True)        # softmax over keys — each row sums to 1
    return w @ v                            # each output = weighted mix of the values`;

  const LOOP_CODE = `model = GPT(vocab_size=65, n_layer=4, n_head=4, n_embd=128, block_size=128)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

for step in range(max_steps):
    xb, yb = get_batch('train')             # (B, T) ints; yb is xb shifted one to the right
    logits = model(xb)                      # (B, T, vocab)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
    opt.zero_grad(set_to_none=True)         # forget last step's gradients
    loss.backward()                         # autograd does what lab 02 does, at scale
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()                              # move every weight a little downhill

    if step % 200 == 0:
        print(step, loss.item(), estimate_val_loss())
        start = torch.zeros((1, 1), dtype=torch.long)   # the newline token
        print(decode(model.generate(start, 200)[0].tolist()))`;

  /* =====================================================================
     Interactive (a): in-browser character-level neural language model
     context of T chars → embedding (E=8) → hidden H tanh → softmax over V
     ===================================================================== */
  function buildTrainer(ctx) {
    const h = ctx.h;
    const B = 32, E = 8;
    const MAX_CHARS = 200000, MAX_VOCAB = 96;
    let T = 3, H = 48, lrExp = -1, temp = 0.8;
    let text = PROSE, pendingText = PROSE;
    let m = null, hist = [], emaHist = [], step = 0, emaLoss = NaN, bestLoss = Infinity;
    let status = 'training', playing = true, needSample = true, plotDirty = true;
    let procCount = 0, procT0 = performance.now(), charsPerSec = 0;

    const [cv, g] = ctx.canvas(720, 250);
    const sampleBox = h('pre', { class: 'code', style: { margin: '10px 0 0', minHeight: '96px', whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: '.85rem', lineHeight: '1.45' } }, '');
    const promptInp = h('input', { type: 'text', placeholder: 'optional prompt, e.g. "the dog"', style: { background: '#172033', color: '#e6ebf5', border: '1px solid #243044', borderRadius: '6px', padding: '6px 8px', fontFamily: 'inherit', width: '100%' } });
    promptInp.addEventListener('input', () => { needSample = true; });
    const rd = ctx.readout();

    function buildModel() {
      // Two guards on user-supplied text: total length (memory) and vocabulary size. The output
      // layer is H×V, so an unbounded V (paste a page of Chinese) would make one training step
      // cost tens of milliseconds and stall the frame. Rare characters collapse into '?'.
      if (text.length > MAX_CHARS) { text = text.slice(0, MAX_CHARS); status = 'text truncated to ' + MAX_CHARS.toLocaleString() + ' characters'; }
      if (text.length < T + 8 || new Set('\n' + text).size < 2) { text = PROSE; status = 'text too short — using the default corpus'; }
      const freq = new Map();
      for (const c of '\n' + text) freq.set(c, (freq.get(c) || 0) + 1);
      let chars = Array.from(freq.keys());
      if (chars.length > MAX_VOCAB) {
        chars.sort((a, b) => freq.get(b) - freq.get(a));
        const keep = new Set(chars.slice(0, MAX_VOCAB - 2));
        keep.add('\n'); keep.add('?');            // '\n' is the padding token, '?' absorbs the rest
        chars = Array.from(keep);
        status = 'vocabulary capped at ' + chars.length + ' characters (rare ones became "?")';
      }
      chars.sort();
      const stoi = {}; chars.forEach((c, i) => { stoi[c] = i; });
      const unk = stoi['?'] != null ? stoi['?'] : stoi['\n'];
      const padded = '\n'.repeat(T) + text;
      const data = new Int32Array(padded.length);
      for (let i = 0; i < padded.length; i++) { const ix = stoi[padded[i]]; data[i] = ix == null ? unk : ix; }
      const V = chars.length, D = T * E;
      const mm = {
        V, T, E, H, D, chars, stoi, unk, data,
        C: new Float64Array(V * E), W1: new Float64Array(D * H), b1: new Float64Array(H), W2: new Float64Array(H * V), b2: new Float64Array(V),
        gC: new Float64Array(V * E), gW1: new Float64Array(D * H), gb1: new Float64Array(H), gW2: new Float64Array(H * V), gb2: new Float64Array(V),
        emb: new Float64Array(D), hact: new Float64Array(H), logits: new Float64Array(V), probs: new Float64Array(V), dh: new Float64Array(H), demb: new Float64Array(D),
        xs: new Int32Array(T),
      };
      for (let i = 0; i < mm.C.length; i++) mm.C[i] = ctx.randn();
      const s1 = (5 / 3) / Math.sqrt(D);              // tanh-friendly (Kaiming-style) init
      for (let i = 0; i < mm.W1.length; i++) mm.W1[i] = ctx.randn() * s1;
      for (let i = 0; i < mm.W2.length; i++) mm.W2[i] = ctx.randn() * 0.01;   // small → loss starts near ln(V)
      mm.nParams = V * E + D * H + H + H * V + V;
      return mm;
    }

    /* forward one example: fills m.emb, m.hact, m.logits */
    function forwardOne(xs) {
      const { E, D, H, V, C, W1, b1, W2, b2, emb, hact, logits } = m;
      for (let j = 0; j < m.T; j++) { const base = xs[j] * E; for (let k = 0; k < E; k++) emb[j * E + k] = C[base + k]; }
      for (let hi = 0; hi < H; hi++) { let s = b1[hi]; for (let d = 0; d < D; d++) s += emb[d] * W1[d * H + hi]; hact[hi] = Math.tanh(s); }
      for (let v = 0; v < V; v++) { let s = b2[v]; for (let hi = 0; hi < H; hi++) s += hact[hi] * W2[hi * V + v]; logits[v] = s; }
    }
    function softmax(logits, probs, V, t) {
      let mx = -Infinity; for (let v = 0; v < V; v++) if (logits[v] > mx) mx = logits[v];
      let sum = 0; for (let v = 0; v < V; v++) { probs[v] = Math.exp((logits[v] - mx) / t); sum += probs[v]; }
      for (let v = 0; v < V; v++) probs[v] /= sum;
    }

    /* one SGD step on a random minibatch of B examples; returns mean loss */
    function trainStep(lr) {
      const { V, D, H, E, T, data, gC, gW1, gb1, gW2, gb2, W1, W2, emb, hact, probs, logits, dh, demb, xs } = m;
      gC.fill(0); gW1.fill(0); gb1.fill(0); gW2.fill(0); gb2.fill(0);
      let loss = 0;
      const maxPos = data.length - T - 1;
      if (maxPos < 0) return 0;                          // cannot happen (buildModel guards) — but never index off the end
      for (let bi = 0; bi < B; bi++) {
        const pos = Math.floor(Math.random() * (maxPos + 1));
        for (let j = 0; j < T; j++) xs[j] = data[pos + j];
        const y = data[pos + T];
        forwardOne(xs);
        softmax(logits, probs, V, 1);
        loss += -Math.log(probs[y] + 1e-12);
        probs[y] -= 1;                                   // dL/dlogits = probs - onehot(y)
        for (let hi = 0; hi < H; hi++) {                 // through W2
          let s = 0; const hv = hact[hi];
          for (let v = 0; v < V; v++) { gW2[hi * V + v] += hv * probs[v]; s += probs[v] * W2[hi * V + v]; }
          dh[hi] = s * (1 - hv * hv);                    // tanh'(x) = 1 - tanh(x)^2
        }
        for (let v = 0; v < V; v++) gb2[v] += probs[v];
        for (let d = 0; d < D; d++) {                    // through W1
          let s = 0; const ed = emb[d];
          for (let hi = 0; hi < H; hi++) { gW1[d * H + hi] += ed * dh[hi]; s += dh[hi] * W1[d * H + hi]; }
          demb[d] = s;
        }
        for (let hi = 0; hi < H; hi++) gb1[hi] += dh[hi];
        for (let j = 0; j < T; j++) { const base = xs[j] * E; for (let k = 0; k < E; k++) gC[base + k] += demb[j * E + k]; }
      }
      // average over the batch, clip the global gradient norm, then SGD
      const pairs = [[m.C, gC], [W1, gW1], [m.b1, gb1], [W2, gW2], [m.b2, gb2]];
      const inv = 1 / B; let norm2 = 0;
      for (const [, gr] of pairs) for (let i = 0; i < gr.length; i++) { gr[i] *= inv; norm2 += gr[i] * gr[i]; }
      const norm = Math.sqrt(norm2);
      const clip = norm > 5 ? 5 / norm : 1;
      const eta = lr * clip;
      for (const [p, gr] of pairs) for (let i = 0; i < p.length; i++) p[i] -= eta * gr[i];
      return loss / B;
    }

    function generate(n, t, prompt) {
      const { T, V, chars, stoi, unk, logits, probs, xs } = m;
      const ctxArr = new Int32Array(T).fill(stoi['\n']);
      const p = (prompt || '').slice(-T);
      for (let i = 0; i < p.length; i++) { const idx = stoi[p[i]]; ctxArr[T - p.length + i] = idx == null ? unk : idx; }
      let out = prompt || '';
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < T; j++) xs[j] = ctxArr[j];
        forwardOne(xs);
        softmax(logits, probs, V, Math.max(0.05, t));
        let r = Math.random(), idx = -1;
        for (let v = 0; v < V; v++) { r -= probs[v]; if (r <= 0) { idx = v; break; } }
        if (idx < 0) idx = V - 1;                // rounding slack, or non-finite probs
        out += chars[idx];
        for (let j = 0; j < T - 1; j++) ctxArr[j] = ctxArr[j + 1];
        ctxArr[T - 1] = idx;
      }
      return out;
    }

    function reset(msg) {
      status = msg || (playing ? 'training' : 'paused');
      m = buildModel();                    // may overwrite `status` with a warning about the text
      hist = []; emaHist = []; step = 0; emaLoss = NaN; bestLoss = Infinity;
      procCount = 0; procT0 = performance.now(); charsPerSec = 0;
      needSample = true; plotDirty = true;
      updateReadout();
    }

    function updateReadout() {
      rd.set({
        step, loss: isFinite(emaLoss) ? emaLoss.toFixed(3) : '—', best: isFinite(bestLoss) ? bestLoss.toFixed(3) : '—',
        'ln(V)': Math.log(m.V).toFixed(2), params: m.nParams.toLocaleString(), vocab: m.V, 'chars/s': Math.round(charsPerSec).toLocaleString(), status,
      });
    }

    function drawLoss() {
      const W = cv.W, HH = cv.H;
      g.clearRect(0, 0, W, HH);
      g.fillStyle = ctx.colors.bg; g.fillRect(0, 0, W, HH);
      const padL = 46, padR = 14, padT = 16, padB = 26;
      const pw = W - padL - padR, ph = HH - padT - padB;
      const lnV = Math.log(m.V);
      const yMax = Math.max(1, lnV * 1.15), yMin = 0;    // max(1, …) so a 2-character vocabulary never divides by ~0
      /* unclamped: a step that overshoots above the axis must run off the top (the clip below
         trims it) instead of being laid flat along the top edge, which would read as a plateau */
      const yOf = v => padT + ph * (1 - (v - yMin) / (yMax - yMin));
      g.strokeStyle = ctx.colors.line; g.lineWidth = 1; g.font = '11px JetBrains Mono, monospace'; g.fillStyle = ctx.colors.muted; g.textAlign = 'right';
      for (let v = 0; v <= yMax; v += 1) { const y = yOf(v); g.beginPath(); g.moveTo(padL, y); g.lineTo(W - padR, y); g.stroke(); g.fillText(v.toFixed(0), padL - 6, y + 4); }
      // uniform-guess baseline
      g.setLineDash([5, 4]); g.strokeStyle = ctx.colors.warn; g.beginPath(); g.moveTo(padL, yOf(lnV)); g.lineTo(W - padR, yOf(lnV)); g.stroke(); g.setLineDash([]);
      g.textAlign = 'left'; g.fillStyle = ctx.colors.warn; g.fillText('ln(V) = ' + lnV.toFixed(2) + '  (uniform guessing)', padL + 6, yOf(lnV) - 5);
      const n = hist.length;
      if (n > 1) {
        g.save(); g.beginPath(); g.rect(padL, padT, pw, ph); g.clip();
        const DOT = 3.5, pwD = pw - DOT - 1;      // the newest point is a dot: leave it room inside the clip
        const cols = Math.min(Math.floor(pwD), n);
        g.strokeStyle = 'rgba(124,156,255,0.25)'; g.lineWidth = 1; g.beginPath();
        for (let c = 0; c < cols; c++) {
          const i0 = Math.floor(c * n / cols), i1 = Math.max(i0 + 1, Math.floor((c + 1) * n / cols));
          let mn = Infinity, mx = -Infinity;
          for (let i = i0; i < i1; i++) { const v = hist[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
          const x = padL + (c + 0.5) / cols * pwD;
          g.moveTo(x, yOf(mx)); g.lineTo(x, yOf(mn) + 0.5);
        }
        g.stroke();
        g.strokeStyle = ctx.colors.accent; g.lineWidth = 2; g.beginPath();
        for (let c = 0; c < cols; c++) {
          const i = Math.min(n - 1, Math.floor((c + 0.5) * n / cols));
          const x = padL + (c + 0.5) / cols * pwD, y = yOf(emaHist[i]);
          if (c === 0) g.moveTo(x, y); else g.lineTo(x, y);
        }
        g.stroke();
        const lx = padL + (cols - 0.5) / cols * pwD, ly = yOf(emaHist[n - 1]);
        g.fillStyle = ctx.colors.accent; g.beginPath(); g.arc(lx, ly, DOT, 0, Math.PI * 2); g.fill();
        g.restore();
      }
      g.fillStyle = ctx.colors.muted; g.textAlign = 'left'; g.fillText('loss (cross-entropy, nats)', padL + 6, padT + 10);
      g.textAlign = 'right'; g.fillText('step ' + step.toLocaleString() + (hist.length ? '  ·  smoothed ' + (emaLoss || 0).toFixed(3) : ''), W - padR, HH - 8);
      if (!playing) { g.fillStyle = ctx.colors.warn; g.textAlign = 'center'; g.font = '600 13px Inter, sans-serif'; g.fillText('paused', W / 2, padT + 14); }
    }

    reset();
    let lastReadout = 0;
    const loop = ctx.loop(() => {
      if (playing) {
        // hard time budget: never spend more than ~5 ms of a 16 ms frame on training
        const budgetEnd = performance.now() + 5;
        let k = 0;
        while (k < 16 && performance.now() < budgetEnd) {
          const lr = Math.pow(10, lrExp);
          const loss = trainStep(lr);
          if (!isFinite(loss)) { reset('NaN detected → reset (lower the learning rate)'); break; }
          step++; k++;
          emaLoss = isNaN(emaLoss) ? loss : emaLoss * 0.97 + loss * 0.03;
          if (emaLoss < bestLoss && step > 50) bestLoss = emaLoss;
          hist.push(loss); emaHist.push(emaLoss);
          procCount += B;
        }
        if (hist.length > 20000) {           // decimate history so drawing stays cheap
          const nh = [], ne = [];
          for (let i = 0; i + 1 < hist.length; i += 2) { nh.push((hist[i] + hist[i + 1]) / 2); ne.push(emaHist[i + 1]); }
          hist = nh; emaHist = ne;
        }
        const now = performance.now();
        if (now - procT0 > 500) { charsPerSec = procCount / ((now - procT0) / 1000); procCount = 0; procT0 = now; }
      }
      if (playing || plotDirty) { plotDirty = false; drawLoss(); }   // paused: nothing changes, don't repaint
      if (performance.now() - lastReadout > 250) { lastReadout = performance.now(); updateReadout(); }
    });

    function sample() {
      sampleBox.textContent = generate(120, temp, promptInp.value);
      needSample = false;
    }
    ctx.interval(() => { if (playing || needSample) sample(); }, 1000);
    sample();

    const playBtn = ctx.button('⏸ Pause', () => {
      playing = !playing; playBtn.textContent = playing ? '⏸ Pause' : '▶ Play'; plotDirty = true;
      if (playing) { procT0 = performance.now(); procCount = 0; status = 'training'; } else { status = 'paused'; charsPerSec = 0; }
      updateReadout();
    }, 'primary');
    const resetBtn = ctx.button('↺ Reset', () => reset('reset'));
    const sampleBtn = ctx.button('✎ Sample now', () => sample());
    const applyBtn = ctx.button('Apply text & reset', () => { text = pendingText; reset('new text'); });
    const lrSl = ctx.slider({ label: 'learning rate (SGD)', min: -2.3, max: 0.3, step: 0.05, value: lrExp, fmt: v => Math.pow(10, v).toFixed(3), onChange: v => { lrExp = v; } });
    const tempSl = ctx.slider({ label: 'temperature (sampling)', min: 0.2, max: 2, step: 0.1, value: temp, onChange: v => { temp = v; needSample = true; } });
    const ctxSel = ctx.select({ label: 'context length (chars)', options: [{ value: '2', label: '2' }, { value: '3', label: '3' }, { value: '4', label: '4' }], value: '3', onChange: v => { T = +v; reset('context = ' + T); } });
    const hidSel = ctx.select({ label: 'hidden size', options: [{ value: '16', label: '16' }, { value: '32', label: '32' }, { value: '48', label: '48' }, { value: '96', label: '96' }], value: '48', onChange: v => { H = +v; reset('hidden = ' + H); } });
    const ta = ctx.textarea({ label: 'training text (edit freely, then "Apply text & reset")', value: PROSE, onChange: v => { pendingText = v; } });
    const preset = ctx.select({ label: 'preset corpus', options: [{ value: 'prose', label: 'simple English prose' }, { value: 'code', label: 'Python-like code' }], value: 'prose', onChange: v => { pendingText = v === 'code' ? PYCODE : PROSE; ta.value = pendingText; text = pendingText; reset('corpus: ' + v); } });

    const body = h('div', {},
      cv,
      h('div', { style: { display: 'flex', gap: '8px', alignItems: 'center', marginTop: '10px', fontSize: '.8rem', color: ctx.colors.muted } }, h('span', { style: { whiteSpace: 'nowrap' } }, 'prompt:'), promptInp),
      h('div', { style: { fontSize: '.75rem', color: ctx.colors.muted, marginTop: '8px', textTransform: 'uppercase', letterSpacing: '.08em' } }, 'sample (120 chars, regenerated every second)'),
      sampleBox);
    const controls = [
      h('div', { class: 'btn-row', style: { margin: 0 } }, playBtn, resetBtn, sampleBtn),
      lrSl, tempSl, ctxSel, hidSel, preset, ta, applyBtn,
    ];
    return ctx.figure(body, 'A character-level neural language model training live in your browser: ' +
      'context of the previous few characters → 8-dim embedding per character → tanh hidden layer → softmax over every character in the text. ' +
      'Forward pass, backward pass and SGD are all written by hand in this page\'s JavaScript (no library). The yellow dashed line is the loss of pure guessing, ln(V). ' +
      'Gradients are clipped to norm 5 and the model resets itself if the loss ever becomes NaN.', controls, rd);
  }

  /* =====================================================================
     Interactive (b): autograd visualiser for L = (a*b + c) * d
     ===================================================================== */
  function buildAutograd(ctx) {
    const h = ctx.h;
    const vals = { a: 2, b: -3, c: 10, d: -2 };
    const nodes = {
      a: { x: 90, y: 62, label: 'a', kind: 'leaf', fd: 0, bd: 3 },
      b: { x: 90, y: 132, label: 'b', kind: 'leaf', fd: 0, bd: 3 },
      c: { x: 90, y: 202, label: 'c', kind: 'leaf', fd: 0, bd: 2 },
      d: { x: 90, y: 272, label: 'd', kind: 'leaf', fd: 0, bd: 1 },
      e: { x: 290, y: 97, label: 'e = a*b', kind: 'op', fd: 1, bd: 2 },
      f: { x: 460, y: 150, label: 'f = e+c', kind: 'op', fd: 2, bd: 1 },
      L: { x: 630, y: 211, label: 'L = f*d', kind: 'out', fd: 3, bd: 0 },
    };
    /* Each edge's local-derivative chip is hand-placed in the empty corridor beside its curve:
       the boxes are 96 wide, so the gaps between the columns are 104/74/74px and a two-line chip
       (symbol above, number below) fits there without touching a node or another chip. */
    const edges = [
      { from: 'a', to: 'e', local: () => vals.b, sym: '∂e/∂a = b', lx: 190, ly: 46 },
      { from: 'b', to: 'e', local: () => vals.a, sym: '∂e/∂b = a', lx: 190, ly: 128 },
      { from: 'e', to: 'f', local: () => 1, sym: '∂f/∂e', lx: 375, ly: 96 },
      { from: 'c', to: 'f', local: () => 1, sym: '∂f/∂c', lx: 190, ly: 172 },
      { from: 'f', to: 'L', local: () => vals.d, sym: '∂L/∂f = d', lx: 545, ly: 148 },
      { from: 'd', to: 'L', local: () => cur.f, sym: '∂L/∂d = f', lx: 360, ly: 254 },
    ];
    const cur = { a: 0, b: 0, c: 0, d: 0, e: 0, f: 0, L: 0 };
    const grad = { a: NaN, b: NaN, c: NaN, d: NaN, e: NaN, f: NaN, L: NaN };
    let phase = 'idle', prog = 0, forwardDone = false, backwardDone = false, dirty = true;
    /* c starts at 10 and its gradient is negative, so descent always wants it to rise:
       the ceiling has to sit above the starting value or Nudge visibly moves three leaves
       out of four. */
    const DUR = 1.8, LO = -5, HI = 12;      // LO/HI must match the sliders' min/max
    const HW = 48;                          // half the width of a node box
    const CHIP_FONT = '10px JetBrains Mono, monospace', CHIP_LH = 12;

    function forwardCompute() {
      cur.a = vals.a; cur.b = vals.b; cur.c = vals.c; cur.d = vals.d;
      cur.e = cur.a * cur.b; cur.f = cur.e + cur.c; cur.L = cur.f * cur.d;
    }
    function backwardCompute() {
      grad.L = 1;
      grad.f = grad.L * cur.d; grad.d = grad.L * cur.f;
      grad.e = grad.f * 1; grad.c = grad.f * 1;
      grad.a = grad.e * cur.b; grad.b = grad.e * cur.a;
    }
    function clearGrads() { for (const k in grad) grad[k] = NaN; backwardDone = false; }

    const [cv, g] = ctx.canvas(720, 330);
    const rd = ctx.readout();
    const fmt = v => (Math.round(v * 100) / 100).toString();

    function draw() {
      const W = cv.W, HH = cv.H;
      g.clearRect(0, 0, W, HH); g.fillStyle = ctx.colors.bg; g.fillRect(0, 0, W, HH);
      const fStage = phase === 'forward' ? prog * 3 : (forwardDone ? 99 : -1);
      const bStage = phase === 'backward' ? prog * 3 : (backwardDone ? 99 : -1);
      let flowingLabel = '';
      const chips = [];                          // edge labels, drawn after the boxes so nothing hides them
      // edges
      for (const ed of edges) {
        const A = nodes[ed.from], Bn = nodes[ed.to];
        const x0 = A.x + HW, y0 = A.y, x1 = Bn.x - HW, y1 = Bn.y;
        const tF = fStage - (Bn.fd - 1);           // 0..1 while the forward pulse travels
        const tB = bStage - (A.bd - 1);            // 0..1 while the backward pulse travels
        const active = (phase === 'forward' && tF > 0 && tF < 1) || (phase === 'backward' && tB > 0 && tB < 1);
        g.strokeStyle = active ? (phase === 'forward' ? ctx.colors.green : ctx.colors.danger) : ctx.colors.line;
        g.lineWidth = active ? 2.5 : 1.5;
        g.beginPath(); g.moveTo(x0, y0); g.bezierCurveTo(x0 + 60, y0, x1 - 60, y1, x1, y1); g.stroke();
        // local derivative label (shown once the backward flow reaches this edge)
        if (bStage >= A.bd - 1 || backwardDone) chips.push(ed);
        if (active) {
          const t = phase === 'forward' ? tF : 1 - tB;
          const s = t, u = 1 - t;
          const px = u * u * u * x0 + 3 * u * u * s * (x0 + 60) + 3 * u * s * s * (x1 - 60) + s * s * s * x1;
          const py = u * u * u * y0 + 3 * u * u * s * y0 + 3 * u * s * s * y1 + s * s * s * y1;
          g.fillStyle = phase === 'forward' ? ctx.colors.green : ctx.colors.danger;
          g.beginPath(); g.arc(px, py, 6, 0, Math.PI * 2); g.fill();
          if (phase === 'backward') flowingLabel = 'grad(' + ed.from + ') += ∂' + ed.to + '/∂' + ed.from + ' × grad(' + ed.to + ') = ' + fmt(ed.local()) + ' × ' + fmt(grad[ed.to]) + ' = ' + fmt(ed.local() * grad[ed.to]);
          else flowingLabel = 'computing ' + Bn.label;
        }
      }
      // nodes
      for (const k in nodes) {
        const n = nodes[k];
        const showVal = fStage >= n.fd || forwardDone;
        const showGrad = (bStage >= n.bd || backwardDone) && !isNaN(grad[k]);
        const col = n.kind === 'leaf' ? ctx.colors.accent : n.kind === 'op' ? ctx.colors.purple : ctx.colors.pink;
        g.fillStyle = '#111827'; g.strokeStyle = col; g.lineWidth = showVal ? 2 : 1;
        rr(n.x - HW, n.y - 26, HW * 2, 52, 8); g.fill(); g.stroke();
        g.fillStyle = col; g.font = '600 12px Inter, sans-serif'; g.textAlign = 'center';
        g.fillText(n.label, n.x, n.y - 10);
        g.font = '11px JetBrains Mono, monospace';
        g.fillStyle = ctx.colors.text; g.fillText('data ' + (showVal ? fmt(cur[k]) : '?'), n.x, n.y + 6);
        g.fillStyle = showGrad ? ctx.colors.danger : ctx.colors.muted; g.fillText('grad ' + (showGrad ? fmt(grad[k]) : '?'), n.x, n.y + 20);
      }
      // edge chips last: an opaque plate keeps the curve from running through the numbers
      g.font = CHIP_FONT; g.textAlign = 'center';
      for (const ed of chips) {
        const l1 = ed.sym, l2 = '= ' + fmt(ed.local());
        const w = Math.max(g.measureText(l1).width, g.measureText(l2).width);
        g.fillStyle = '#141b28'; g.strokeStyle = 'rgba(251,113,133,0.35)'; g.lineWidth = 1;
        g.fillRect(ed.lx - w / 2 - 6, ed.ly - 9, w + 12, CHIP_LH + 13);
        g.strokeRect(ed.lx - w / 2 - 6, ed.ly - 9, w + 12, CHIP_LH + 13);
        g.fillStyle = ctx.colors.danger;
        g.fillText(l1, ed.lx, ed.ly); g.fillText(l2, ed.lx, ed.ly + CHIP_LH);
      }
      g.fillStyle = ctx.colors.muted; g.font = '12px Inter, sans-serif'; g.textAlign = 'left';
      g.fillText(flowingLabel || (backwardDone ? 'done: every leaf now knows how much L changes if it changes' : forwardDone ? 'forward done — press Backward to send dL/dL = 1 back through the graph' : 'press Forward'), 20, HH - 14);
    }
    /* Traced side by side rather than with arcTo: an arcTo path is described by its corner
       targets, so the "outline" cuts diagonally across the middle of the box it is meant to
       surround — harmless on screen, but it is not the shape this box actually is. */
    function rr(x, y, w, hh, r) {
      g.beginPath();
      g.moveTo(x + r, y); g.lineTo(x + w - r, y); g.quadraticCurveTo(x + w, y, x + w, y + r);
      g.lineTo(x + w, y + hh - r); g.quadraticCurveTo(x + w, y + hh, x + w - r, y + hh);
      g.lineTo(x + r, y + hh); g.quadraticCurveTo(x, y + hh, x, y + hh - r);
      g.lineTo(x, y + r); g.quadraticCurveTo(x, y, x + r, y);
      g.closePath();
    }

    function updateRd() {
      rd.set({ a: fmt(vals.a), b: fmt(vals.b), c: fmt(vals.c), d: fmt(vals.d), L: forwardDone ? fmt(cur.L) : '?',
        'dL/da': backwardDone ? fmt(grad.a) : '?', 'dL/db': backwardDone ? fmt(grad.b) : '?', 'dL/dc': backwardDone ? fmt(grad.c) : '?', 'dL/dd': backwardDone ? fmt(grad.d) : '?' });
    }
    ctx.loop((dt) => {
      if (phase !== 'idle') {
        prog += dt / DUR;
        if (prog >= 1) { prog = 1; if (phase === 'forward') forwardDone = true; else backwardDone = true; phase = 'idle'; updateRd(); }
        dirty = true;
      }
      if (dirty) { dirty = false; draw(); }     // idle and unchanged: no repaint
    });

    /* step 0.01 so a nudged value always lands exactly on the slider's grid */
    const sliders = ['a', 'b', 'c', 'd'].map(k => ctx.slider({ label: k, min: LO, max: HI, step: 0.01, value: vals[k], onChange: v => { vals[k] = v; forwardDone = false; clearGrads(); phase = 'idle'; dirty = true; updateRd(); } }));
    const fwdBtn = ctx.button('▶ Forward', () => { forwardCompute(); clearGrads(); phase = 'forward'; prog = 0; forwardDone = false; dirty = true; });
    const bwdBtn = ctx.button('◀ Backward', () => { if (!forwardDone) { forwardCompute(); forwardDone = true; } backwardCompute(); phase = 'backward'; prog = 0; backwardDone = false; dirty = true; }, 'primary');
    const nudgeBtn = ctx.button('Nudge leaves by −0.05·grad', () => {
      if (!backwardDone) { forwardCompute(); forwardDone = true; backwardCompute(); backwardDone = true; }
      ['a', 'b', 'c', 'd'].forEach((k, i) => {
        // clamp to the slider range: L = (a·b+c)·d is unbounded below, so unclamped descent
        // would run the numbers off the screen in a dozen clicks
        vals[k] = ctx.clamp(Math.round((vals[k] - 0.05 * grad[k]) * 100) / 100, LO, HI);
        sliders[i].value = vals[k];
      });
      forwardCompute(); forwardDone = true; backwardCompute(); backwardDone = true; phase = 'idle'; dirty = true; updateRd();
    });
    updateRd();
    return ctx.figure(cv, 'The expression graph for L = (a·b + c)·d. <b>Forward</b> fills the data fields left to right. <b>Backward</b> starts with grad(L) = 1 and multiplies by each edge\'s local derivative on the way back: that product is the chain rule, and the red numbers on the edges are exactly what <code class="inline">_backward()</code> computes in lab 02. ' +
      '<b>Nudge</b> moves every leaf a little against its gradient and recomputes: L goes down, which is all gradient descent ever does.', [h('div', { class: 'btn-row', style: { margin: 0 } }, fwdBtn, bwdBtn, nudgeBtn), ...sliders], rd);
  }

  /* =====================================================================
     Interactive (c): loss-at-init sanity checker
     ===================================================================== */
  function buildInitChecker(ctx) {
    const h = ctx.h;
    let logV = Math.log10(65), logStd = -1;     // vocab 65 (Shakespeare chars), logit std 0.1
    const S = 22;                                  // sample points along the curve
    const POOL = 32768, MAXN = 8192;               // fixed pool of N(0,1) draws, reused for every V
    let V = 65, baseN = 0, base = [], curve = [];
    const [cv, g] = ctx.canvas(720, 240);
    const rd = ctx.readout();
    const stds = []; for (let i = 0; i < S; i++) stds.push(Math.pow(10, -2 + 3 * i / (S - 1)));  // 0.01 … 10
    /* One pool of standard normals, drawn once. Every V and every σ reuses it, which makes the
       curve a smooth, deterministic function of σ — so the red dot always lands exactly on it
       instead of jittering around it, and moving a slider never re-rolls the dice. */
    const normals = new Float32Array(POOL);
    for (let i = 0; i < POOL; i++) normals[i] = ctx.randn();

    /* Expected cross-entropy of a fresh model whose V logits are N(0, σ²), averaged over every
       possible target: E[logsumexp(z)] − mean(z). We average over as many independent draws as
       the pool allows (hundreds when V is small, four when it is large). For V above MAXN we
       score MAXN logits and scale the sum by V/MAXN — exact for any vocabulary up to 8,192,
       within ~2% above it for σ ≤ 3, and always under a millisecond. */
    function lossRaw(std, n) {
      const draws = Math.max(1, Math.floor(POOL / n));
      let tot = 0;
      for (let s = 0; s < draws; s++) {
        const off = s * n;
        let mx = -Infinity;
        for (let i = 0; i < n; i++) { const z = normals[off + i] * std; if (z > mx) mx = z; }
        let sum = 0, mean = 0;
        for (let i = 0; i < n; i++) { const z = normals[off + i] * std; sum += Math.exp(z - mx); mean += z; }
        tot += mx + Math.log(sum) - mean / n;     // logsumexp minus the mean target logit
      }
      return tot / draws;
    }
    function lossFor(std) { const n = Math.min(V, MAXN); return lossRaw(std, n) + Math.log(V / n); }
    function recompute() {
      V = Math.max(2, Math.round(Math.pow(10, logV)));
      const n = Math.min(V, MAXN);
      if (n !== baseN) { baseN = n; base = stds.map(s => lossRaw(s, n)); }  // only when the sample size changes
      const corr = Math.log(V / n);
      curve = base.map(l => l + corr);            // a bigger vocabulary just lifts the whole curve
      draw();
    }
    function draw() {
      const W = cv.W, HH = cv.H;
      g.clearRect(0, 0, W, HH); g.fillStyle = ctx.colors.bg; g.fillRect(0, 0, W, HH);
      const padL = 46, padR = 16, padT = 18, padB = 30, pw = W - padL - padR, ph = HH - padT - padB;
      const lnV = Math.log(V);
      const cur = lossFor(Math.pow(10, logStd));
      const yMax = Math.max(lnV * 2.2, cur * 1.1, 2);
      /* no clamp here: a loss above the top of the axis must leave the plot (the clip below trims
         it), because flattening it onto the top edge would draw a plateau the model does not have */
      const xOf = s => padL + (Math.log10(s) + 2) / 3 * pw, yOf = l => padT + ph * (1 - l / yMax);
      g.strokeStyle = ctx.colors.line; g.lineWidth = 1; g.font = '11px JetBrains Mono, monospace'; g.fillStyle = ctx.colors.muted;
      [0.01, 0.1, 1, 10].forEach((s, i) => {
        const x = xOf(s); g.beginPath(); g.moveTo(x, padT); g.lineTo(x, padT + ph); g.stroke();
        /* the outermost ticks sit on the plot edges, so their labels hang off the canvas if centred */
        g.textAlign = i === 0 ? 'left' : i === 3 ? 'right' : 'center';
        g.fillText('σ = ' + s, x, HH - 10);
      });
      const stepY = yMax > 12 ? 5 : yMax > 6 ? 2 : 1;
      for (let l = 0; l <= yMax; l += stepY) { const y = yOf(l); g.beginPath(); g.moveTo(padL, y); g.lineTo(W - padR, y); g.stroke(); g.textAlign = 'right'; g.fillText(String(l), padL - 6, y + 4); }
      g.setLineDash([5, 4]); g.strokeStyle = ctx.colors.warn; g.beginPath(); g.moveTo(padL, yOf(lnV)); g.lineTo(W - padR, yOf(lnV)); g.stroke(); g.setLineDash([]);
      g.fillStyle = ctx.colors.warn; g.textAlign = 'left'; g.fillText('ln(V) = ' + lnV.toFixed(2) + '  (the loss of an honest "I have no idea")', padL + 6, yOf(lnV) - 5);
      g.save(); g.beginPath(); g.rect(padL, padT, pw, ph); g.clip();
      g.strokeStyle = ctx.colors.accent; g.lineWidth = 2; g.beginPath();
      curve.forEach((l, i) => { const x = xOf(stds[i]), y = yOf(l); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); });
      g.stroke();
      g.restore();
      const cx = xOf(Math.pow(10, logStd)), cy = yOf(cur);
      g.strokeStyle = ctx.colors.danger; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(cx, padT); g.lineTo(cx, padT + ph); g.stroke(); g.setLineDash([]);
      g.fillStyle = cur > lnV * 1.15 ? ctx.colors.danger : ctx.colors.green; g.beginPath(); g.arc(cx, cy, 6, 0, Math.PI * 2); g.fill();
      g.fillStyle = ctx.colors.text; g.font = '600 12px Inter, sans-serif'; g.textAlign = cx > W / 2 ? 'right' : 'left';
      /* for small σ the dot sits exactly on the ln(V) line, where this label would be printed on
         top of the ln(V) one — drop it below the dot whenever the two are that close */
      const nearLnV = Math.abs(cy - yOf(lnV)) < 22;
      g.fillText('measured init loss ' + cur.toFixed(2), cx + (cx > W / 2 ? -10 : 10), cy + (nearLnV ? 24 : -10));
      g.fillStyle = ctx.colors.muted; g.font = '11px Inter, sans-serif'; g.textAlign = 'left'; g.fillText('initial loss vs. the std-dev σ of the random logits (log scale)', padL + 6, padT + 10);
      const verdict = cur > lnV * 1.15 ? 'confidently wrong at init → shrink the last layer\'s weights (or zero its bias)' : cur < lnV * 0.9 ? 'below ln(V): only possible if the targets are not uniform — check for leakage' : 'healthy: about ln(V), the model starts out humble';
      rd.set({ 'vocab V': V.toLocaleString(), 'ln(V)': lnV.toFixed(3), 'logit σ': Math.pow(10, logStd).toFixed(3), 'init loss': cur.toFixed(3), verdict });
    }
    const vSl = ctx.slider({ label: 'vocabulary size V', min: Math.log10(2), max: Math.log10(200000), step: 0.02, value: logV, fmt: v => Math.round(Math.pow(10, v)).toLocaleString(), onChange: v => { logV = v; scheduleRecompute(); } });
    const sSl = ctx.slider({ label: 'std-dev of random logits σ', min: -2, max: 1, step: 0.05, value: logStd, fmt: v => Math.pow(10, v).toFixed(3), onChange: v => { logStd = v; draw(); } });
    let pending = false, alive = true;
    ctx.onCleanup(() => { alive = false; });      // a queued frame must not touch a detached canvas
    function scheduleRecompute() { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; if (alive) recompute(); }); }
    const setV = (v) => { logV = v; vSl.value = v; recompute(); };
    const presets = h('div', { class: 'btn-row', style: { margin: 0 } },
      ctx.button('chars (65)', () => setV(Math.log10(65))),
      ctx.button('GPT-2 BPE (50,257)', () => setV(Math.log10(50257))),
      ctx.button('Llama 3 (128,256)', () => setV(Math.log10(128256))));
    recompute();
    return ctx.figure(cv, 'A freshly initialised model with random logits of std-dev σ. With tiny σ every class gets equal probability and the loss is exactly ln(V): the model admits it knows nothing. With large σ it is confidently wrong about random targets and the loss is far higher. The first number your training script prints should be close to the yellow line; if it is not, you have found your first bug before wasting an hour of GPU time.', [vSl, sSl, presets], rd);
  }

  /* =====================================================================
     Chapter registration
     ===================================================================== */

  function wrapLines(gc, text, maxW) {
    const words = String(text).split(' '); const out = []; let line = '';
    for (const w of words) {
      const t = line ? line + ' ' + w : w;
      if (line && gc.measureText(t).width > maxW) { out.push(line); line = w; } else line = t;
    }
    if (line) out.push(line);
    return out;
  }
  function wrapText(gc, text, x, y, maxW, lh) {
    wrapLines(gc, text, maxW).forEach((ln, i) => gc.fillText(ln, x, y + i * lh));
  }

  /* ------------------------------------------------------------------ */
  /* Interactive: read a broken loss curve and name the bug              */
  /* ------------------------------------------------------------------ */
  function buildLossDoctor(ctx) {
    const [cv, g] = ctx.canvas(720, 360);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    const V = 65, LN_V = Math.log(V);
    const CASES = [
      {
        name: 'flat as a board',
        curve: (t) => LN_V - 0.02 * t + 0.03 * Math.sin(t * 9),
        causes: ['learning rate far too low, or the optimizer never sees the gradients', 'learning rate far too high', 'not enough training data', 'the model is too small'],
        answer: 0,
        why: 'A loss pinned at ln(V) means the weights are barely moving. The classic causes are a forgotten <code>optimizer.step()</code>, a missing <code>loss.backward()</code>, parameters never passed to the optimizer, or a learning rate some orders of magnitude too small. Check the gradient norms: if they are zero, it is wiring, not tuning.',
      },
      {
        name: 'shoots to NaN',
        curve: (t) => t < 0.25 ? LN_V - 1.6 * t : LN_V + Math.pow((t - 0.25) * 3.4, 2) * 4,
        causes: ['learning rate too low', 'learning rate too high, or exploding gradients', 'the dataset is too small', 'wrong loss function'],
        answer: 1,
        why: 'The loss drops, then rockets. Each step is overshooting the valley and landing higher on the far side — chapter 2\'s failure mode, at full scale. Drop the learning rate by 3×, add gradient clipping, and check for a division by something that can be zero.',
      },
      {
        name: 'starts far too high',
        curve: (t) => 9.4 - 5.0 * t * t,
        causes: ['the data is shuffled wrong', 'the final layer is initialised too confidently', 'the batch size is too small', 'the model needs more layers'],
        answer: 1,
        why: 'At step 0 a healthy model should sit at about ln(V) = ' + LN_V.toFixed(2) + ' for ' + V + ' classes, because it should be maximally unsure. Starting at 9 means the output layer is making confident wrong predictions at random initialisation. Scale its weights down — this is exactly what the initialisation checker below measures.',
      },
      {
        name: 'train falls, validation rises',
        curve: (t) => LN_V - 3.3 * Math.pow(t, 0.55),
        curve2: (t) => LN_V - 2.6 * Math.pow(t, 0.5) + Math.max(0, (t - 0.42)) * 4.2,
        causes: ['the learning rate is too high', 'overfitting — stop early, add data, or regularise', 'the gradients are vanishing', 'the tokenizer is broken'],
        answer: 1,
        why: 'Textbook overfitting, and exactly the curve you watched in chapter 3. The model is memorising the training set. Stop at the validation minimum, add more data, add weight decay or dropout, or make the model smaller.',
      },
      {
        name: 'cannot memorise 32 examples',
        curve: (t) => LN_V - 0.9 * Math.pow(t, 0.7),
        causes: ['the model needs more training time', 'wired wrong: shifted targets or a dead gradient', 'the learning rate is too low', 'the batch size is too large'],
        answer: 1,
        why: 'This is the most valuable check in the list. Train on <b>one batch of 32</b> and the loss must go essentially to zero — memorising 32 examples takes no cleverness at all. If it plateaus, the bug is structural: targets shifted by one, a mask hiding the answer, or a tensor detached from the graph. No amount of tuning fixes it.',
      },
    ];
    let idx = 0, picked = -1;

    const nextBtn = ctx.button('Next symptom →', () => { idx = (idx + 1) % CASES.length; picked = -1; }, 'primary');
    const opts = [0, 1, 2, 3].map(i => ctx.button('answer ' + (i + 1), () => { picked = i; }));
    const ro = ctx.readout();

    ctx.loop(() => {
      g.clearRect(0, 0, cv.W, cv.H);
      const c = CASES[idx];
      const P = { x: 55, y: 44, w: 300, h: 170 };
      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('symptom: ' + c.name, P.x, 28);
      g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
      const px = (t) => P.x + t * P.w;
      const py = (v) => P.y + P.h - ctx.clamp((v - 0) / 11, 0, 1) * P.h;
      /* the ln(V) reference line every healthy run starts on */
      g.setLineDash([4, 4]); g.strokeStyle = C.warn; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(P.x, py(LN_V)); g.lineTo(P.x + P.w, py(LN_V)); g.stroke();
      g.setLineDash([]);
      g.font = MONO; g.fillStyle = C.warn;
      g.fillText('ln(' + V + ') = ' + LN_V.toFixed(2) + '  ← a healthy start', P.x + 4, py(LN_V) - 6);

      const draw = (fn, col) => {
        g.strokeStyle = col; g.lineWidth = 2.5; g.beginPath();
        for (let i = 0; i <= 100; i++) {
          const t = i / 100; let v = fn(t);
          if (!isFinite(v) || v > 11) v = 11;
          i ? g.lineTo(px(t), py(v)) : g.moveTo(px(t), py(v));
        }
        g.stroke();
      };
      draw(c.curve, C.accent);
      if (c.curve2) { draw(c.curve2, C.danger); }
      g.font = MONO; g.fillStyle = C.accent; g.fillText('train', P.x + 8, P.y + P.h - 10);
      if (c.curve2) { g.fillStyle = C.danger; g.fillText('validation', P.x + 8, P.y + P.h - 26); }
      g.fillStyle = C.muted; g.fillText('steps →', P.x + P.w / 2 - 20, P.y + P.h + 18);

      /* the options */
      const TX = 385;
      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('what is wrong?', TX, 28);
      c.causes.forEach((txt, i) => {
        const y = 44 + i * 44;
        const right = i === c.answer, chosen = i === picked;
        g.fillStyle = picked < 0 ? '#141b28' : right ? 'rgba(56,217,169,0.18)' : chosen ? 'rgba(251,113,133,0.18)' : '#141b28';
        g.fillRect(TX, y, 300, 38);
        g.strokeStyle = picked < 0 ? C.line : right ? C.green : chosen ? C.danger : C.line;
        g.lineWidth = 1.5; g.strokeRect(TX, y, 300, 38);
        g.font = MONO; g.fillStyle = C.muted; g.fillText(String(i + 1), TX + 8, y + 22);
        g.font = FONT; g.fillStyle = picked < 0 ? C.muted : right ? C.green : C.text;
        /* the box holds two lines; if an option ever outgrows it, say so visibly
           rather than deleting the end of the sentence */
        const lines = wrapLines(g, txt, 262);
        if (lines.length > 2) { lines.length = 2; lines[1] = lines[1].replace(/\s*\S*$/, '') + ' …'; }
        lines.forEach((ln, j) => g.fillText(ln, TX + 26, y + 16 + j * 15));
      });

      if (picked >= 0) {
        g.font = 'bold ' + FONT;
        g.fillStyle = picked === c.answer ? C.green : C.warn;
        g.fillText(picked === c.answer ? '✓ Correct.' : '✗ Not that one — here is what it actually is:', 55, 250);
        g.font = FONT; g.fillStyle = C.muted;
        wrapText(g, c.why.replace(/<[^>]+>/g, ''), 55, 272, 610, 17);
      } else {
        g.font = FONT; g.fillStyle = C.muted;
        wrapText(g, 'Read the curve, then press one of the answer buttons. Every training run that fails, fails quietly — there is no stack trace for "the loss is flat", so the shape of the curve is most of the evidence you get.', 55, 250, 610, 17);
      }
      ro.set({ symptom: c.name, 'of': idx + 1 + ' / ' + CASES.length, answered: picked < 0 ? '—' : (picked === c.answer ? 'correct' : 'try again') });
    });

    return ctx.figure(cv,
      'Five real failure modes, and the diagnosis each curve points to. The dashed yellow line is ln(V) — where a healthy run starts, because a model that knows nothing should be maximally unsure across V classes. Most debugging is reading how a curve departs from that line: pinned to it means nothing is updating, well above it means the output layer is over-confident at initialisation, and a validation curve peeling upward means memorisation.',
      [nextBtn, ...opts], ro);
  }

  /* ------------------------------------------------------------------ */
  /* Interactive: the shapes flowing through a tiny GPT                  */
  /* ------------------------------------------------------------------ */
  function buildShapeFlow(ctx) {
    const [cv, g] = ctx.canvas(720, 360);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    let B = 32, T = 8, D = 64, H = 4, V = 65;
    const bSl = ctx.slider({ label: 'batch B', min: 1, max: 64, step: 1, value: 32, onChange: (v) => { B = v; } });
    const tSl = ctx.slider({ label: 'context T', min: 2, max: 64, step: 1, value: 8, onChange: (v) => { T = v; } });
    /* step 4, not 8: the caption tells the reader to try d = 100, which a step of 8 cannot reach */
    const dSl = ctx.slider({ label: 'width d', min: 8, max: 256, step: 4, value: 64, onChange: (v) => { D = v; } });
    const hSl = ctx.slider({ label: 'heads', min: 1, max: 12, step: 1, value: 4, onChange: (v) => { H = v; } });
    const ro = ctx.readout();

    ctx.loop(() => {
      g.clearRect(0, 0, cv.W, cv.H);
      const divides = D % H === 0;
      const headDim = D / H;
      const rows = [
        { n: 'idx  (the token ids you fed in)', s: '(' + B + ', ' + T + ')', ok: true },
        { n: 'token embedding table lookup', s: '(' + B + ', ' + T + ', ' + D + ')', ok: true },
        { n: '+ position embedding', s: '(' + B + ', ' + T + ', ' + D + ')', ok: true, note: 'broadcast over the batch' },
        { n: 'split into heads', s: '(' + B + ', ' + H + ', ' + T + ', ' + (divides ? headDim : '?') + ')', ok: divides, note: divides ? 'd ÷ heads = ' + headDim : 'd is not divisible by heads' },
        { n: 'attention scores  q @ kᵀ', s: '(' + B + ', ' + H + ', ' + T + ', ' + T + ')', ok: divides, note: 'the T×T grid — grows as T²' },
        { n: 'weighted values, heads merged', s: '(' + B + ', ' + T + ', ' + D + ')', ok: divides },
        { n: 'MLP  d → 4d → d', s: '(' + B + ', ' + T + ', ' + D + ')', ok: true, note: 'hidden layer is ' + (4 * D) },
        { n: 'final projection to the vocabulary', s: '(' + B + ', ' + T + ', ' + V + ')', ok: true },
        { n: 'cross-entropy wants it flattened', s: '(' + (B * T) + ', ' + V + ') vs targets (' + (B * T) + ',)', ok: true, note: 'the reshape everyone forgets' },
      ];
      /* three fixed columns: the widest name, the widest shape and the widest note each have to
         clear the next column, and the last note has to finish inside 720px */
      const NAME_X = 30, SHAPE_X = 252, NOTE_X = 496;
      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('every tensor shape in one forward pass', NAME_X, 24);
      let y = 40;
      rows.forEach(r => {
        g.font = FONT; g.fillStyle = r.ok ? C.muted : C.danger;
        g.fillText(r.n, NAME_X, y + 13);
        g.font = 'bold ' + MONO; g.fillStyle = r.ok ? C.accent : C.danger;
        g.fillText(r.s, SHAPE_X, y + 13);
        if (r.note) { g.font = MONO; g.fillStyle = r.ok ? C.line : C.danger; g.fillText(r.note, NOTE_X, y + 13); }
        y += 25;
      });
      g.font = 'bold 15px Inter, system-ui, sans-serif';
      g.fillStyle = divides ? C.green : C.danger;
      g.fillText(divides ? 'shapes line up' : 'd must be divisible by the number of heads — this run would crash', NAME_X, 280);
      const attnCells = B * H * T * T;
      g.font = MONO; g.fillStyle = C.muted;
      g.fillText('attention matrix holds ' + attnCells.toLocaleString() + ' numbers', NAME_X, 301);
      g.font = FONT; g.fillStyle = C.muted;
      wrapText(g, 'Double the context and that number quadruples. It is the first thing to shrink when you run out of memory.', NAME_X, 321, 640, 16);
      ro.set({ 'B,T,d,heads': B + ',' + T + ',' + D + ',' + H, 'head dim': divides ? headDim : 'INVALID', 'attn cells': attnCells.toLocaleString() });
    });

    return ctx.figure(cv,
      'The single most common thing that stops a from-scratch transformer from running is a shape mismatch, and the single most common mismatch is a width that is not divisible by the number of heads. Drag <b>width d</b> to 100 with 8 heads and watch the row go red. The last row is the other classic: cross-entropy wants the logits flattened to (B·T, V) against targets of shape (B·T,), and forgetting that reshape produces an error message that tells you almost nothing useful.',
      [bSl, tSl, dSl, hSl], ro);
  }

  ZTA.registerChapter({
    id: '13-build-it-yourself',
    num: 13,
    part: 'IV',
    title: 'Build it yourself: from one neuron to a tiny GPT',
    tagline: 'Eight short programs, each a working model you wrote by hand. The last one writes Shakespeare-ish, and you will understand every line.',
    render(root, ctx) {
      const h = ctx.h;
      const lab = (f) => '<code class="inline">labs/' + f + '</code>';

      root.append(
        ctx.callout('tryit', '🖐 Do this first — watch a language model learn to spell, from nothing',
          'This is a real neural language model training in your browser right now. Nothing is pre-computed.<br>' +
          '<b>1.</b> Just watch the sample box for twenty seconds. It starts as noise, then spaces appear at word-like intervals, then real words.<br>' +
          '<b>2.</b> Watch the loss start near <b>ln(V)</b> — the value a model that knows nothing must have — and fall.<br>' +
          '<b>3.</b> Read the parameter count in the readout — it is under three thousand. That is a model learning English spelling from scratch, in your browser, in under a minute.'),
        buildTrainer(ctx),
        ctx.p('Every idea in the previous twelve chapters is in those few hundred lines. The rest of this chapter is how to write them yourself.'),

        ctx.p('You have spent twelve chapters watching models learn. Sliders moved, loss curves fell, decision boundaries bent. But every one of those demos was written by someone else. There is a particular moment, and most people who work on AI remember theirs, when you type a training loop yourself, run it, and watch numbers you fully understand turn into behaviour you never explicitly programmed. After that moment the whole field looks different: less like magic, more like plumbing you could fix.'),
        ctx.p('Richard Feynman left a line on his blackboard: <em>"What I cannot create, I do not understand."</em> Andrej Karpathy adopted it as the motto for his from-scratch tutorials, and it is the motto of this chapter. The question we answer is practical. What is the shortest sequence of programs, each small enough to read in one sitting, that takes you from a single artificial neuron to a working GPT, and what do you do after that?'),
        ctx.p('The answer is a ladder of eight labs in the <code class="inline">labs/</code> folder of this repository. Every rung is a complete, runnable model. Together they are under two thousand lines of code. Read them, break them, fix them. But before you open a terminal, train a language model right here, in the next section, and watch it learn to spell.'),

        ctx.section('First, watch a language model learn to spell',
          ctx.p('The model below is the smallest thing that deserves the name <em>neural language model</em>. It is the architecture Yoshua Bengio\'s group proposed in 2003, and the one Karpathy uses in his "makemore" series. It reads the previous three characters and predicts the next one. That is the entire task, and it is the same task GPT does, with two differences: GPT reads thousands of tokens instead of three characters, and it uses a transformer instead of one hidden layer.'),
          ctx.p('Here is the whole forward pass. Each of the three context characters is looked up in an <em>embedding table</em> and becomes a vector of 8 numbers. The three vectors are glued into one vector of 24. A layer of 48 <em>tanh</em> neurons squashes that into 48 numbers. A final layer turns those into one score per character in the vocabulary, and a <em>softmax</em> turns the scores into probabilities. The loss is <em>cross-entropy</em>: minus the log of the probability the model gave to the character that actually came next. Backpropagation computes how every weight should move to raise that probability, and SGD moves it.'),
          ctx.p('Count the parameters for a text with 30 distinct characters: 30 × 8 in the embedding table, 24 × 48 + 48 in the hidden layer, 48 × 30 + 30 in the output layer. That is 2,910 numbers. GPT-3 has 175 billion. The recipe is identical; only the numbers are bigger.'),
          ctx.callout('key', 'What you just saw is the whole field', 'Loss starts at ln(V) because the model starts ignorant. It falls fast as it learns which characters are common, then slowly as it learns which character follows which. It never reaches zero, because English is not fully predictable from three characters. Everything in the rest of this course, from transformers to RLHF, is about making that curve go lower, on more data, with more context. The mechanics you can now see are not a simplification; they are the thing itself.'),
        ),

        ctx.section('Setting up',
          ctx.p('You need Python 3.10 or newer and about ten minutes. A GPU is not required for any lab; the tiny GPT trains on a laptop CPU in minutes at the small setting. From the repository root:'),
          ctx.code(`python -m venv .venv
# Windows:            .venv\\Scripts\\activate
# macOS / Linux:      source .venv/bin/activate
pip install -r requirements.txt          # numpy, torch (CPU build is fine), matplotlib
python labs/01_perceptron.py             # each lab is one file; run it, then read it`, 'bash'),
          ctx.p('The CPU build of PyTorch is enough. If you have an NVIDIA GPU, install the CUDA build from pytorch.org and lab 06 will pick it up automatically; on Apple silicon the <code class="inline">mps</code> backend works too. Details, expected outputs and the exact commands are in <code class="inline">labs/README.md</code>.'),
        ),

        ctx.section('The ladder: eight labs, eight ideas',
          ctx.p('Each lab isolates one idea and is written to be read top to bottom. For every rung, here is what to read first, what to change, what to observe, and the bug you will probably write. Do not skip the "change" step. Reading code teaches you what it does; breaking it teaches you why it is written that way.'),

          ctx.sub('01 · Perceptron (pure Python, no libraries) — ' + lab('01_perceptron.py'),
            ctx.p('Rosenblatt\'s 1958 machine in forty lines: a weighted sum, a threshold, and an update rule that nudges the weights toward every example it gets wrong. This is the ancestor of every neuron in every model that follows.'),
            ctx.ul([
              '<b>Read:</b> the update rule. It is one line: <code class="inline">w += lr * (y - y_hat) * x</code>. Everything else is bookkeeping.',
              '<b>Change:</b> the learning rate (it barely matters here, which is itself a lesson) and the dataset. Then try to make it learn XOR.',
              '<b>Observe:</b> on linearly separable data it converges in a handful of passes. On XOR it never settles. This is the problem Minsky and Papert used to sink the field in 1969 (chapter 14).',
              '<b>Common bug:</b> forgetting the bias term, or mixing 0/1 labels with a −1/+1 update rule.',
            ])),

          ctx.sub('02 · An autograd engine — ' + lab('02_autograd.py'),
            ctx.p('This is the lab that turns backpropagation from a formula into a tool. A <code class="inline">Value</code> class wraps a number and remembers which two values made it and how. Calling <code class="inline">backward()</code> on the final loss walks that graph in reverse and fills in every gradient. It is a hundred lines, and it is <b>exactly</b> what PyTorch does when you call <code class="inline">loss.backward()</code>, minus the tensors and the C++.'),
            ctx.code(VALUE_CODE, 'python'),
            ctx.ul([
              '<b>Read:</b> <code class="inline">__mul__</code> and <code class="inline">backward()</code>. Notice the closure: each op stores a tiny function that knows the local derivative. Notice the topological sort: a node\'s gradient is only complete after all its consumers have reported in.',
              '<b>Change:</b> add <code class="inline">tanh</code>, <code class="inline">exp</code> and <code class="inline">__pow__</code>. Then build a single neuron, then a layer, then check your gradients against finite differences: nudge one input by 1e-6 and compare the change in output with what backward() claims.',
              '<b>Observe:</b> gradients <i>accumulate</i> with <code class="inline">+=</code>. A value used twice gets two contributions. If you write <code class="inline">=</code> instead, the second path silently overwrites the first.',
              '<b>Common bug:</b> not zeroing the gradients between training steps, so each step adds to the last. Loss goes up and you have no idea why.',
            ]),
            ctx.callout('tryit', 'Try it', 'Set a=2, b=−3, c=10, d=−2 (the defaults, from Karpathy\'s micrograd lecture). Press Forward, then Backward. Follow the red pulse: grad(L)=1 arrives at f and d; the edge f→L multiplies it by d = −2, so grad(f) = −2; the + node passes −2 unchanged to e and c; the edge a→e multiplies by b, so grad(a) = 6. Now press Nudge three times and watch L fall. Then set b = 0 and run backward again: why is grad(a) now zero?'),
            buildAutograd(ctx)),

          ctx.sub('03 · A multilayer perceptron trained with your own engine — ' + lab('03_mlp.py'),
            ctx.p('Stack the neurons from lab 02 into layers, feed them 2-D points from two interleaved moons, and train with the loop you will use for the rest of your life: forward, loss, zero the grads, backward, update.'),
            ctx.ul([
              '<b>Read:</b> the training loop. It is five lines and it never changes shape again, all the way to GPT.',
              '<b>Change:</b> hidden width (4 → 16 → 64), learning rate (0.01 → 0.1 → 1.0), the dataset (moons → circles → XOR).',
              '<b>Observe:</b> the decision boundary bending around the moons; the loss curve going ragged when the learning rate is too high; a width-2 network failing where width-16 succeeds.',
              '<b>Common bug:</b> the sign of the update. <code class="inline">w += lr * w.grad</code> climbs the loss. It is astonishing how often this one happens.',
            ])),

          ctx.sub('04 · A BPE tokenizer — ' + lab('04_tokenizer_bpe.py'),
            ctx.p('Before a language model sees text, the text is chopped into tokens. Byte-pair encoding does this by repeatedly merging the most frequent adjacent pair. The algorithm is a loop of ten lines. Its consequences (why models are bad at counting letters, why code costs more tokens than prose) are felt in every product you use.'),
            ctx.ul([
              '<b>Read:</b> the merge loop: count adjacent pairs, merge the most frequent, record the merge, repeat.',
              '<b>Change:</b> <code class="inline">--vocab-size</code>, which sets 256 byte tokens plus N merges — so 300 buys 44 merges, not 300. Train on prose, then on Python.',
              '<b>Observe:</b> "the " becoming a single token early; numbers being split into odd pieces; the tokens-per-character ratio falling as the vocabulary grows.',
              '<b>Common bug:</b> encoding new text with the merges applied in a different order than they were learned. Order is the tokenizer.',
            ])),

          ctx.sub('05 · Attention in NumPy — ' + lab('05_attention.py'),
            ctx.p('One function, and it is the heart of the transformer. Every token asks a question (its <em>query</em>), every token advertises what it holds (its <em>key</em>), and each token\'s output is a weighted average of everyone\'s <em>values</em>, weighted by how well query matches key. The causal mask stops a token from peeking at the future.'),
            ctx.code(ATTN_CODE, 'python'),
            ctx.ul([
              '<b>Read:</b> the mask line and the softmax axis. Both are one-liners and both are where bugs live.',
              '<b>Change:</b> delete the mask (the model can now read the answer during training). Remove the 1/√d scaling (the softmax saturates and gradients vanish). Split into several heads.',
              '<b>Observe:</b> the T×T weight matrix plotted as a heatmap: lower-triangular <i>including</i> the diagonal — a token always attends to itself — with every row summing to one.',
              '<b>Common bug:</b> masking with 0 instead of −∞ (a zero score still gets probability), or softmax over the wrong axis.',
            ])),

          ctx.sub('06 · A tiny GPT in PyTorch, trained on Shakespeare — ' + lab('06_tiny_gpt.py'),
            ctx.p('Everything comes together: a token embedding, a positional embedding, a stack of blocks (attention, then a small MLP, each wrapped in a residual connection and a LayerNorm), and a final layer that predicts the next character. It trains on the one-megabyte <code class="inline">labs/data/tinyshakespeare.txt</code>, 65 distinct characters, the same file Karpathy\'s nanoGPT uses.'),
            ctx.code(LOOP_CODE, 'python'),
            ctx.p('Here is what to expect as it trains, so you know whether things are working. The loss is a ladder and every rung has a look. At step 0 it should be about 4.17, which is ln(65). By a few hundred steps it is near 3.0: the model has learned letter frequencies, so the samples are full of e, t, a and spaces, and still noise.'),
            ctx.p(' At about 2.5 the nonsense becomes pronounceable — "the wither sould hin". Near 2.0 you get real short words, line breaks in sensible places, and <code class="inline">NAME:</code> speaker headers.'),
            ctx.p(' The default run (four layers, 128 dimensions, four heads, context 128, roughly 0.8 million parameters, 5,000 steps) lands between 1.6 and 1.8: character names, sentence rhythm, the occasional stage direction. Below 1.5 you need a bigger model — six layers, 384 dimensions, context 256 — and the output reads like a drunk Elizabethan: fluent nonsense in perfect form.'),
            ctx.p(' Under about 0.9 on this much text you are no longer learning English, you are memorising Shakespeare.'),
            ctx.p('Time: <code class="inline">--quick</code> (two layers, 64 dimensions, 300 steps) finishes in a minute or two on a laptop CPU and is enough to watch the loss fall off the ln(65) line. The default run takes ten to fifteen minutes on that same CPU and about two minutes on a GPU. The six-layer version really wants a GPU: roughly three minutes there, several hours on CPU. Run that one overnight, or rent a card for an hour.'),
            ctx.ul([
              '<b>Read:</b> the <code class="inline">Block</code> class, then <code class="inline">generate()</code>: the model is called once per new character, with the context cropped to the last <code class="inline">block_size</code> tokens.',
              '<b>Change:</b> <code class="inline">n_layer</code>, <code class="inline">n_embd</code>, <code class="inline">block_size</code>, learning rate, dropout. Train on a different text: your favourite novel, your own chat logs, a code base.',
              '<b>Observe:</b> train loss vs. validation loss drifting apart as the model starts memorising; samples at temperature 0.5 vs. 1.5; how much worse context 8 is than context 64.',
              '<b>Common bug:</b> targets not shifted by one position; the causal mask missing (training loss drops to nearly zero because the model reads the answer); forgetting <code class="inline">torch.no_grad()</code> when sampling (slow and memory-hungry, not wrong).',
            ])),

          ctx.sub('07 · Q-learning — ' + lab('07_q_learning.py'),
            ctx.p('A tabular agent in a grid world. No neural network yet; a table of numbers, one per (state, action), updated by the Bellman rule until the numbers point the way to the goal. This is the algorithm behind DQN, which is the algorithm behind Atari, which is the grandparent of the RL used to train reasoning models.'),
            ctx.ul([
              '<b>Read:</b> the update: <code class="inline">Q[s,a] += alpha * (r + gamma * max(Q[s2]) - Q[s,a])</code>. The bracket is the <em>temporal-difference error</em>: what you got versus what you expected.',
              '<b>Change:</b> ε (exploration), γ (how much the future matters), α (learning rate), and the reward for stepping (try a small negative one).',
              '<b>Observe:</b> the Q-table printed as arrows: it becomes a gradient flowing toward the goal. With ε too small the agent commits early to a mediocre route.',
              '<b>Common bug:</b> bootstrapping from the terminal state (its future value must be zero), and never decaying ε so the agent wanders forever.',
            ])),

          ctx.sub('08 · Toy RLHF — ' + lab('08_toy_rlhf.py'),
            ctx.p('The whole post-training pipeline from chapter 11 at toy scale: a small policy model, a reward model trained on pairs of outputs where one was preferred, and a policy-gradient step that raises the reward while a KL penalty keeps the policy near its starting point.'),
            ctx.ul([
              '<b>Read:</b> the three stages in order: supervised fine-tune, reward model on preferences (the Bradley–Terry loss), policy update with a KL term.',
              '<b>Change:</b> the KL coefficient β. Set it to zero and watch <em>reward hacking</em>: the policy finds a degenerate output the reward model adores and the text becomes garbage. Set it high and the policy barely moves.',
              '<b>Observe:</b> reward going up while sample quality goes down is the single most important phenomenon in alignment; you can produce it on a laptop.',
              '<b>Common bug:</b> forgetting to freeze the reference model, so the KL penalty is measured against a moving target and does nothing.',
            ])),
        ),

        ctx.section('The bug that stops every from-scratch transformer',
          ctx.p('Before the debugging ritual, the failure that comes first: the model will not even run. In a from-scratch transformer that is almost always a tensor shape.'),
          ctx.callout('tryit', '🖐 Try this',
            '<b>1.</b> Drag <b>width d</b> to 100 while <b>heads</b> is 8. The split-into-heads row turns red — d must divide evenly by the number of heads, and this is the crash almost everyone hits first.<br>' +
            '<b>2.</b> Now drag <b>context T</b> from 8 to 64 and watch the attention-matrix count. It grows with T², which is why context is the first thing to shrink when you run out of memory.<br>' +
            '<b>3.</b> Read the last row. Cross-entropy wants the logits flattened to (B·T, V); forgetting that reshape gives an error message that tells you almost nothing.'),
          buildShapeFlow(ctx),
        ),

        ctx.section('How to debug a model that will not learn',
          ctx.callout('tryit', '🖐 Try this — diagnose five broken training runs',
            '<b>1.</b> Read each curve against the dashed yellow line at ln(V), which is where a healthy run <i>starts</i>.<br>' +
            '<b>2.</b> Pick an answer, then read the explanation whether you were right or not.<br>' +
            '<b>3.</b> Pay particular attention to the last one — "cannot memorise 32 examples" is the single most valuable check in the list, and no amount of tuning fixes what it catches.'),
          buildLossDoctor(ctx),

          ctx.p('Every training run that fails, fails quietly. There is no stack trace for "the loss is flat". The remedy is a fixed ritual of sanity checks, cheap enough to run every single time, most of them borrowed from Karpathy\'s <i>A Recipe for Training Neural Networks</i>.'),
          ctx.ol([
            '<b>Check the loss at initialisation.</b> With V classes and a humble model it should be ≈ ln(V). If it is much higher, your last layer is too confident at random; scale its weights down. The interactive below shows why.',
            '<b>Overfit a single batch.</b> Take one batch of 32 examples and train on only that. The loss must go essentially to zero. If it cannot memorise 32 examples, something is wired wrong: shifted targets, a mask, a broken gradient.',
            '<b>Grad-check with finite differences.</b> For a few parameters, compute (loss(w+ε) − loss(w−ε)) / 2ε with ε = 1e-5 and compare with the gradient backprop reports. Agreement to four or five digits means the backward pass is right.',
            '<b>Sweep the learning rate.</b> Try 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 for a few hundred steps each and plot. The best one is usually a factor of three below where the loss explodes. AdamW wants roughly 3e-4 for a large transformer and about 1e-3 for a small one like lab 06; plain SGD wants a hundred times more than either.',
            '<b>Look at the data.</b> Decode a batch and print it. Half of all bugs are a shuffled label, a tokenizer mismatch, or validation data leaking into training.',
            '<b>Fix the seed.</b> <code class="inline">torch.manual_seed(1337)</code>. If two runs of the same code differ, the difference is not your change.',
          ]),
          ctx.callout('tryit', 'Try it', 'Press <b>chars (65)</b> and set σ to 0.1: the loss sits on the yellow line at 4.18 ≈ ln(65). Healthy. Now drag σ up to about 3 and the loss roughly doubles, to around 8, even though the model has learned precisely nothing — it is confidently wrong. Push σ to 10 and it passes 20. Now press <b>GPT-2 BPE</b>: ln(V) jumps to 10.82, and <b>Llama 3</b> takes it to 11.76. When the first line of a real GPT-2 training run prints a loss near 10.9, you now know that is the number it is supposed to print, not a bug — and that a first line reading 15 means the last layer is initialised far too hot.'),
          buildInitChecker(ctx),
          ctx.callout('example', 'What this looks like in a real run', 'The first three lines of a healthy nanoGPT run on Shakespeare read something like <code class="inline">step 0: train loss 4.2825, val loss 4.2822</code>, then <code class="inline">step 250: train loss 2.4914</code>, then <code class="inline">step 500: train loss 2.1240</code>. Three numbers, and an experienced person has already checked three things: the first is ln(65) so the init is sane; train and val agree so nothing has leaked; and the drop is fast but not instant, so the targets are shifted correctly. When someone glances at a log and says "that looks wrong", this is what they are doing.'),
        ),

        ctx.section('Scaling the ladder',
          ctx.p('The eight labs end at a model with a few million parameters. Frontier models have a few trillion. The ladder continues; it just leaves your laptop.'),
          ctx.sub('Reproduce GPT-2',
            ctx.p('Karpathy\'s <a href="https://github.com/karpathy/nanoGPT" target="_blank">nanoGPT</a> is lab 06 grown up: the same GPT class, plus data loading, mixed precision and multi-GPU support. Point it at a web-text dataset (OpenWebText, or FineWeb) and it reproduces the 124-million-parameter GPT-2 on a rented cloud machine.'),
            ctx.p(' In <a href="https://github.com/karpathy/llm.c" target="_blank">llm.c</a>, his C/CUDA rewrite, the same reproduction on 10 billion tokens takes about an hour and a half on eight A100s and costs on the order of twenty dollars; the 1.5-billion-parameter GPT-2 XL takes about a day on eight H100s and a few hundred dollars.'),
            ctx.p(' In 2019 GPT-2 was the most capable language model on Earth, and OpenAI trained it on 32 TPU v3 chips for a week — about $43,000 of compute. By 2026 Karpathy\'s nanochat reaches the same capability in three hours on one eight-GPU node, for roughly $73. He puts that at a 600× fall in seven years: the price of a fixed capability is dropping about 2.5× a year, and has not stopped.'),
          ),
          ctx.sub('Fine-tune an open model',
            ctx.p('Pretraining from scratch is the expensive part; you almost never need to. Open-weight models (Llama, Qwen, Gemma, Mistral, DeepSeek) are pretrained on trillions of tokens and released for free.'),
            ctx.p(' <em>LoRA</em> (low-rank adaptation) freezes the base model and trains small adapter matrices, about 1% of the parameters, so a 7B model fine-tunes on a single consumer GPU. <em>QLoRA</em> quantises the frozen base to 4 bits so a 70B model fits on one 48 GB card.'),
            ctx.p(' Hugging Face\'s <a href="https://github.com/huggingface/trl" target="_blank">TRL</a> library gives you SFT, reward modelling, DPO and GRPO trainers that are lab 08 at production quality; <a href="https://github.com/unslothai/unsloth" target="_blank">Unsloth</a> makes the same runs two to five times faster on modest hardware. A weekend and fifty dollars gets you a model that beats any frontier model at one narrow task you define, because you have data they do not.'),
          ),
          ctx.sub('Build an eval',
            ctx.p('The most underrated project. Write two hundred questions with verifiable answers in a domain you know, run every model you can reach against them with a script, and publish the table. <a href="https://github.com/EleutherAI/lm-evaluation-harness" target="_blank">lm-evaluation-harness</a> gives you the plumbing. A good eval is more valuable to the field than a mediocre model, and it teaches you where models actually fail rather than where Twitter says they do.'),
          ),
          ctx.sub('Build an agent harness',
            ctx.p('Take a frontier model through its API (Claude or OpenAI), give it tools (a shell, a file system, a browser), and write the loop from chapter 12: model proposes an action, your code executes it, the result goes back in the context, repeat. The loop is thirty lines. The hard part, and the part that will teach you the most about what models can and cannot do, is the harness around it: sandboxing, timeouts, retries, deciding when to stop, and measuring success on a task suite you build yourself.'),
          ),
        ),

        ctx.callout('example', 'Where this shows up', 'Every major lab\'s training code is a scaled-up version of lab 06: the same forward pass, the same cross-entropy, the same AdamW, distributed across thousands of GPUs. PyTorch\'s autograd is lab 02 with tensors instead of scalars. The GPT-2 tokenizer that ChatGPT descended from is lab 04 with 50,000 merges. And "overfit a single batch first" is the sentence most often said inside those labs when a run misbehaves.'),
        ctx.callout('history', 'Who wrote this ladder', 'The from-scratch tradition in modern deep learning owes most to Andrej Karpathy, whose 2015 blog post <i>The Unreasonable Effectiveness of Recurrent Neural Networks</i> came with a 100-line character-level model that thousands of people trained on Shakespeare, and whose 2022–2023 <i>Neural Networks: Zero to Hero</i> videos (micrograd, makemore, nanoGPT) established the exact sequence these labs follow. The scalar autograd engine idea goes back further: automatic differentiation was worked out in the 1960s and 1970s (Wengert, Linnainmaa) long before anyone called it backpropagation.'),

        ctx.section('Why this matters for modern AI',
          ctx.p('The people who push the frontier are not the ones who have read the most papers. They are the ones who can sit down with a failing training run and find the bug, who can look at a loss curve and know whether the learning rate is wrong, who can implement a new idea in an afternoon because the machinery is already in their hands.'),
          ctx.p(' That skill is built by the ladder, not by reading about it. Every technique in chapters 10 to 12 (mixture-of-experts, RLHF, speculative decoding) was first a small script someone wrote to see if it would work. Chapter 15 will lay out how one person gets from here to that frontier; this chapter is the part you cannot skip.'),
        ),

        ctx.quiz([
          { q: 'A character-level model with 65 distinct characters prints a loss of 4.2 on its first step. What does that tell you?', options: ['The learning rate is too high', 'Something is wrong: the loss should start near zero', 'It is healthy: ln(65) ≈ 4.17, the loss of uniform guessing', 'The model is overfitting'], answer: 2, explain: 'A model that knows nothing should spread its probability evenly, giving loss = ln(V). A much higher initial loss means the model is confidently wrong at random, usually because the last layer\'s weights are too large.' },
          { q: 'In the Value class, why do the backward functions use += rather than = when writing gradients?', options: ['It is faster', 'A value used in several places receives a gradient contribution from each place, and they must add up', 'To avoid dividing by zero', 'Because gradients are always positive'], answer: 1, explain: 'If a is used twice (say L = a*a), the total derivative is the sum of the two paths. Overwriting with = keeps only one and silently produces a wrong gradient.' },
          { q: 'You delete the causal mask from your GPT and the training loss immediately drops to nearly zero. What happened?', options: ['The model got smarter', 'Each position can now see the token it is supposed to predict, so it copies the answer', 'The learning rate is too low', 'The tokenizer broke'], answer: 1, explain: 'Without the mask, position t attends to position t+1, which is exactly its target. The model "predicts" by looking. Suspiciously good training loss is a bug until proven otherwise.' },
          { q: 'What is the point of "overfit a single batch" as a debugging step?', options: ['It makes the final model better', 'A correct pipeline must be able to memorise 32 examples; if it cannot, the bug is in the wiring, not the data or the scale', 'It measures generalisation', 'It is a way to choose the batch size'], answer: 1, explain: 'It isolates the machinery. Memorising one batch is trivially possible for any working model, so failure to do so pinpoints a structural bug before you spend hours on a full run.' },
          { q: 'What does LoRA do?', options: ['Trains every weight of a large model with a lower learning rate', 'Freezes the base model and trains small low-rank adapter matrices, so fine-tuning fits on modest hardware', 'Compresses the model after training', 'Replaces attention with a cheaper operation'], answer: 1, explain: 'LoRA adds a small trainable low-rank update to chosen weight matrices while the original weights stay frozen. Roughly 1% of the parameters train, memory drops dramatically, and quality is close to full fine-tuning.' },
        ]),

        ctx.section('Go deeper',
          ctx.ul([
            '<a href="https://www.youtube.com/watch?v=VMj-3S1tku0" target="_blank">Karpathy — The spelled-out intro to neural networks and backpropagation (micrograd)</a>: lab 02, narrated, in 2.5 hours.',
            '<a href="https://www.youtube.com/watch?v=kCc8FmEb1nY" target="_blank">Karpathy — Let\'s build GPT: from scratch, in code, spelled out</a>: lab 06, line by line.',
            '<a href="http://karpathy.github.io/2019/04/25/recipe/" target="_blank">A Recipe for Training Neural Networks</a>: the debugging ritual this chapter borrows from.',
            '<a href="https://github.com/karpathy/nanoGPT" target="_blank">nanoGPT</a> and <a href="https://github.com/karpathy/llm.c" target="_blank">llm.c</a>: reproduce GPT-2 for tens of dollars.',
            '<a href="https://github.com/huggingface/trl" target="_blank">Hugging Face TRL</a>, <a href="https://github.com/unslothai/unsloth" target="_blank">Unsloth</a> and <a href="https://github.com/EleutherAI/lm-evaluation-harness" target="_blank">lm-evaluation-harness</a>: fine-tune and evaluate open models.',
            '<a href="https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf" target="_blank">Bengio et al. 2003 — A Neural Probabilistic Language Model</a>: the paper behind the model you trained in your browser.',
          ]),
        ),
      );
    },
  });
})();
