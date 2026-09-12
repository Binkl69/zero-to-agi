/* Zero → AGI · Chapter 14 · Eighty years, in one scroll
   DESIGN RULE: the reader scrolls the whole history and finds the two winters themselves before
   any prose runs.
   Interactives, in order: the zoomable timeline with the compute curve; a prediction game
   (experts were decades wrong in both directions); state-of-the-art by year across five fields
   converging into one architecture; the compute-doubling explorer; and the collapsing price of a
   fixed capability. */
(function () {
  const TAGS = [
    { id: 'theory', label: 'theory', color: '#7c9cff' },
    { id: 'architecture', label: 'architecture', color: '#a78bfa' },
    { id: 'hardware', label: 'hardware', color: '#fb923c' },
    { id: 'data', label: 'data', color: '#38d9a9' },
    { id: 'product', label: 'product', color: '#f472b6' },
    { id: 'milestone', label: 'milestone', color: '#fbbf24' },
    { id: 'winter', label: 'winter', color: '#fb7185' },
  ];
  const TAGCOL = {}; TAGS.forEach(t => { TAGCOL[t.id] = t.color; });

  /* t: decimal year for placement. scale: parameter count. flop: training compute (FLOP). est: flagged as estimate. */
  const ENTRIES = [
    { t: 1943.9, year: '1943', tag: 'theory', title: 'McCulloch and Pitts describe the first artificial neuron',
      desc: 'A neurophysiologist and a homeless teenage logician in Chicago showed that a simple unit which sums its inputs and fires when the sum crosses a threshold can compute any logical function. Their paper, <i>A Logical Calculus of the Ideas Immanent in Nervous Activity</i>, was almost unreadable and enormously influential: it gave von Neumann the vocabulary for the stored-program computer and gave everyone after them the idea that thinking might be arithmetic.' },
    { t: 1950.75, year: '1950', tag: 'theory', title: 'Turing asks "Can machines think?" and proposes the imitation game',
      desc: 'In <i>Computing Machinery and Intelligence</i>, Alan Turing replaced an unanswerable question with a test: if a machine, conversing by text, cannot be told apart from a person, it is thinking for every practical purpose. He predicted machines would pass it by around 2000, discussed how a "child machine" might be educated rather than programmed, and anticipated most of the objections still argued about today. He died four years later.' },
    { t: 1956.5, year: '1956', tag: 'milestone', title: 'The Dartmouth workshop coins "artificial intelligence"',
      desc: 'John McCarthy, Marvin Minsky, Nathaniel Rochester and Claude Shannon proposed a two-month, ten-person summer study on the conjecture that "every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it". The workshop produced no results but named the field, gathered its founders, and set the tone of confident optimism that would take decades to earn.' },
    { t: 1958.5, year: '1958', tag: 'architecture', title: 'Rosenblatt\'s Perceptron: a machine that learns from examples', scale: 400,
      desc: 'Frank Rosenblatt, a psychologist at Cornell, built a device with a 20×20 grid of photocells whose connections adjusted themselves as it was shown examples, the first learning neural network in hardware. The Navy funded it; the New York Times reported it was expected to walk, talk, see and become conscious. It could learn to tell simple shapes apart, and nothing else, but the update rule you wrote in lab 01 is his.' },
    { t: 1966.0, year: '1966', tag: 'product', title: 'ELIZA, the first chatbot',
      desc: 'Joseph Weizenbaum\'s ELIZA at MIT reflected users\' sentences back as questions in the style of a Rogerian therapist, using a few dozen pattern-matching rules. Weizenbaum was disturbed to find that people, including his own secretary, confided in it and attributed understanding to it. The "ELIZA effect", our readiness to see a mind behind fluent text, is the oldest fact about how humans meet language machines.' },
    { t: 1969.5, year: '1969', tag: 'theory', title: 'Minsky and Papert\'s "Perceptrons" proves the limits of one layer',
      desc: 'The book showed mathematically that a single-layer perceptron cannot learn XOR or tell whether a shape is connected. Multi-layer networks could, but nobody knew how to train them, and the authors doubted it was possible. Funding for neural networks collapsed for fifteen years. The book was correct; the conclusion people drew from it was not.' },
    { t: 1974.0, end: 1980.0, year: '1974–80', tag: 'winter', title: 'The first AI winter',
      desc: 'Machine translation had failed to deliver, the Lighthill Report (1973) told the British government that AI had achieved nothing of its grand promises, and DARPA cut open-ended research funding. Toy problems that worked in the lab collapsed on real data: the "combinatorial explosion" of search. The field learned that demos are not products, a lesson it would need to relearn.' },
    { t: 1980.5, year: '1980s', tag: 'product', title: 'Expert systems boom: AI as a business',
      desc: 'DEC\'s XCON configured computer orders using thousands of hand-written if-then rules and saved the company tens of millions of dollars a year; corporations created "AI departments"; Japan launched its Fifth Generation Computer project in 1982 and the US and UK responded with programmes of their own. Knowledge was typed in by hand, rule by rule. It worked, until the rules had to change.' },
    { t: 1986.8, year: '1986', tag: 'theory', title: 'Backpropagation, popularised',
      desc: 'Rumelhart, Hinton and Williams\' Nature paper <i>Learning representations by back-propagating errors</i> showed that multi-layer networks can be trained by the chain rule, and that their hidden layers learn useful internal representations. The idea had been derived several times before (Linnainmaa 1970, Werbos 1974), but this was the paper that made it a movement. It answered Minsky and Papert seventeen years late.' },
    { t: 1987.5, end: 1993.5, year: '1987–93', tag: 'winter', title: 'The second AI winter',
      desc: 'Specialised Lisp machines were killed by cheap general-purpose workstations, expert systems turned out to be brittle and expensive to maintain, and the Fifth Generation project ended without its goals. "AI" became a word grant applicants avoided; the same work continued under names like "machine learning" and "informatics". Neural network research survived in a few labs in Toronto, Montreal and Paris.' },
    { t: 1989.9, year: '1989–98', tag: 'architecture', title: 'LeNet: convolutional networks read handwritten digits', scale: 60000,
      desc: 'Yann LeCun at Bell Labs trained a network with convolutional layers by backpropagation to read handwritten ZIP codes, and by 1998 LeNet-5 (about 60,000 parameters) was reading a meaningful share of the cheques deposited in the United States. Convolution gave the network the built-in assumption that a pattern matters wherever it appears. It was the right architecture, twenty years before the hardware and data that would make it famous.' },
    { t: 1997.35, year: '1997', tag: 'milestone', title: 'Deep Blue defeats Garry Kasparov',
      desc: 'IBM\'s chess machine beat the world champion 3½–2½ in New York, evaluating around 200 million positions per second with hand-tuned evaluation functions and no learning at all. It was a triumph of search and engineering rather than of intelligence, and it is a useful reminder that much of what we once called AI was brute force with a good opening book.' },
    { t: 1997.85, year: '1997', tag: 'architecture', title: 'LSTM: a memory cell for recurrent networks',
      desc: 'Sepp Hochreiter and Jürgen Schmidhuber\'s Long Short-Term Memory added gates that let a recurrent network decide what to remember and what to forget, curing the vanishing-gradient problem Hochreiter had diagnosed in his 1991 thesis. Ignored for a decade, LSTMs powered speech recognition, translation and Siri-era assistants by the mid-2010s, and were the dominant sequence model until the transformer.' },
    { t: 1999.8, year: '1999', tag: 'hardware', title: 'GeForce 256: the first "GPU"',
      desc: 'NVIDIA coined the term for a chip that did the geometry and lighting of 3-D games in hardware. Its purpose was faster Quake. But a graphics processor is, at heart, a machine for doing the same arithmetic on thousands of numbers at once, which is precisely what a neural network needs. Nobody in AI was paying attention yet.' },
    { t: 2006.5, year: '2006', tag: 'theory', title: '"Deep learning": Hinton\'s deep belief nets',
      desc: 'Geoffrey Hinton, Simon Osindero and Yee-Whye Teh showed that a deep network could be trained one layer at a time as a stack of restricted Boltzmann machines, then fine-tuned. The pretraining trick was later abandoned, but the rebrand stuck: "deep learning" replaced the winter-tainted "neural networks", and the Canadian Institute for Advanced Research kept a small group funded through the lean years.' },
    { t: 2007.45, year: '2007', tag: 'hardware', title: 'CUDA lets ordinary programmers use GPUs for maths',
      desc: 'NVIDIA released a C-like language for running general computation on its graphics cards. Scientists began using $500 gaming cards as personal supercomputers. Within five years graduate students in Toronto would train image classifiers on two of them, and the entire economics of the field would tilt toward whoever could buy the most GPUs.' },
    { t: 2009.45, year: '2009', tag: 'data', title: 'ImageNet: 3.2 million labelled images, on the way to 14 million',
      desc: 'Fei-Fei Li\'s team at Princeton and Stanford used Amazon Mechanical Turk to label photographs into thousands of categories, at a time when most vision datasets held a few thousand images. The 2009 paper describes 3.2 million images across 5,247 categories and calls the project about 10% complete; it reached 14 million over the following years. Colleagues told her it was a waste of time. From 2010 the ImageNet challenge gave the field a shared scoreboard, and a scoreboard is what turns a research community into a race.' },
    { t: 2012.75, year: '2012', tag: 'milestone', title: 'AlexNet wins ImageNet and the deep-learning era begins', scale: 6.1e7, flop: 4.7e17,
      desc: 'Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton trained a 60-million-parameter convolutional network on two GeForce GTX 580 gaming cards for a week and cut the ImageNet error rate from 26% to 15%. Every other entry used hand-designed features. Within two years every entry was a deep network, and every large technology company was hiring Hinton\'s students. This is the moment the five ingredients first arrived together.' },
    { t: 2013.1, year: '2013', tag: 'theory', title: 'word2vec: meaning as geometry',
      desc: 'Tomas Mikolov and colleagues at Google trained a tiny network to predict neighbouring words and found the learned vectors encoded meaning: king − man + woman ≈ queen. Embeddings, the idea that a concept can be a point in space and similarity can be a distance, became the foundation of every language model since.' },
    { t: 2013.95, year: '2013', tag: 'milestone', title: 'DQN: one network learns to play Atari from pixels',
      desc: 'DeepMind, a small London start-up, combined Q-learning (lab 07) with a convolutional network and trained it to play dozens of Atari games from raw screen pixels and the score alone, several at superhuman level. Google bought the company weeks later for around $500 million. Deep reinforcement learning was born.' },
    { t: 2014.45, year: '2014', tag: 'architecture', title: 'GANs: two networks in a forgery contest',
      desc: 'Ian Goodfellow\'s generative adversarial network pitted a generator that makes fake images against a discriminator that spots them. The idea reportedly came to him in a Montreal pub and he coded it that night. GANs produced the first convincing synthetic faces and dominated image generation until diffusion models overtook them around 2021.' },
    { t: 2014.7, year: '2014', tag: 'architecture', title: 'Sequence-to-sequence learning, and the first attention',
      desc: 'Sutskever, Vinyals and Le showed that one LSTM could read a sentence into a vector and another could write out its translation. Bahdanau, Cho and Bengio then let the decoder look back at every input word and weight them, the first attention mechanism, and translation quality jumped. Google Translate switched to neural networks in 2016. Attention was about to eat everything else.' },
    { t: 2015.9, year: '2015', tag: 'architecture', title: 'ResNet: 152 layers deep, with skip connections', scale: 6e7,
      desc: 'Kaiming He and colleagues at Microsoft Research Asia found that adding a shortcut around every pair of layers, so each block learns only a correction to its input, made networks of a hundred-plus layers trainable. ResNet won ImageNet with better-than-human accuracy. The same year gave us batch normalisation and the Adam optimiser; the residual connection lives inside every transformer.' },
    { t: 2016.2, year: '2016', tag: 'milestone', title: 'AlphaGo beats Lee Sedol 4–1',
      desc: 'Go was supposed to be a decade away: its board has more positions than atoms in the universe, and intuition seemed essential. DeepMind\'s AlphaGo combined deep networks trained on human games with reinforcement learning and tree search. Move 37 of game two, a play no professional would have made, was the first time a machine showed something that looked like taste. Two hundred million people watched.' },
    { t: 2017.37, year: '2017', tag: 'hardware', title: 'Google\'s TPU v2: chips designed only for neural networks',
      desc: 'Google revealed it had been running its own tensor processing units in production since 2015 and announced the second generation, which could train as well as run models, connected in "pods" of hundreds of chips. For the first time an AI lab owned its silicon end to end. NVIDIA\'s data-centre GPUs and Google\'s TPUs have alternated at the frontier of training hardware ever since.' },
    { t: 2017.45, year: '2017', tag: 'architecture', title: '"Attention Is All You Need": the Transformer', scale: 2.13e8, flop: 2.3e19, est: true,
      desc: 'Eight researchers at Google Brain threw away recurrence entirely and built a translation model from stacked self-attention and small feed-forward layers. It trained in a fraction of the time because every position could be processed in parallel, which meant it could use GPUs fully, which meant it could scale. Every frontier language model, and most image, audio and protein models, is a descendant of this eleven-page paper.' },
    { t: 2018.45, year: '2018', tag: 'milestone', title: 'GPT-1: pretrain on text, then fine-tune', scale: 1.17e8,
      desc: 'OpenAI\'s Alec Radford trained a 117-million-parameter transformer decoder to predict the next word on 7,000 unpublished books, then fine-tuned it on each benchmark and beat specialised systems on most of them. The paper was modest; the recipe was the one still used today: generative pretraining, then adaptation.' },
    { t: 2018.78, year: '2018', tag: 'milestone', title: 'BERT: bidirectional pretraining sweeps the benchmarks', scale: 3.4e8,
      desc: 'Google\'s BERT trained a transformer encoder to fill in masked words, reading both left and right context, and set new records on eleven language-understanding tasks at once. It went into Google Search within a year. For a while it looked like encoders would win; then scale favoured the generative decoders.' },
    { t: 2019.12, year: '2019', tag: 'milestone', title: 'GPT-2: "too dangerous to release"', scale: 1.5e9, flop: 1.5e21, est: true,
      desc: 'Ten times larger than GPT-1 and trained on 40 GB of web pages, GPT-2 wrote coherent multi-paragraph articles with no fine-tuning at all, including a famous fake news story about unicorns in the Andes. OpenAI initially withheld the full 1.5-billion-parameter model, citing misuse, and released it in stages over nine months. It is the model you can now reproduce in an afternoon (chapter 13).' },
    { t: 2020.06, year: '2020', tag: 'theory', title: 'Scaling laws: loss is a smooth function of size, data and compute',
      desc: 'Jared Kaplan and colleagues at OpenAI showed that language-model loss falls as a clean power law in parameters, dataset size and compute, across seven orders of magnitude, with no sign of stopping. This turned "make it bigger" from a hunch into a forecast, and justified spending sums nobody had spent before on a single training run.' },
    { t: 2020.37, year: '2020', tag: 'hardware', title: 'NVIDIA A100',
      desc: 'The Ampere data-centre GPU, with 40 then 80 GB of high-bandwidth memory and dedicated tensor cores for mixed-precision matrix multiplication, became the workhorse of the GPT-3-to-GPT-4 era. Clusters of thousands of A100s, connected by NVLink and InfiniBand, are what "a supercomputer" meant in AI from 2020 to 2023.' },
    { t: 2020.4, year: '2020', tag: 'milestone', title: 'GPT-3: 175 billion parameters and in-context learning', scale: 1.75e11, flop: 3.1e23,
      desc: 'A hundred times larger than GPT-2, trained on roughly 300 billion tokens for a few million dollars of compute, GPT-3 could do tasks from a handful of examples placed in its prompt, with no gradient updates: translation, arithmetic, code, poetry. Nobody had designed that ability in. It was the first model offered as a paid API, and the first that made outsiders take AGI timelines seriously.' },
    { t: 2020.45, year: '2020', tag: 'architecture', title: 'DDPM: diffusion models learn to denoise',
      desc: 'Jonathan Ho, Ajay Jain and Pieter Abbeel showed that a network trained to remove a little noise from an image, applied repeatedly starting from pure static, generates images of a quality that rivalled GANs, with far more stable training. Within two years diffusion powered DALL·E 2, Midjourney and Stable Diffusion; within four, video.' },
    { t: 2021.02, year: '2021', tag: 'milestone', title: 'CLIP and DALL·E: text and images in one space',
      desc: 'OpenAI released both on the same day. CLIP learned from 400 million image–caption pairs to place pictures and text in a shared embedding space, so it could classify images it had never been trained on from a description alone. DALL·E generated images from text prompts ("an armchair in the shape of an avocado"). Together they made multimodal AI a mainstream research direction.' },
    { t: 2021.5, year: '2021', tag: 'product', title: 'Codex and GitHub Copilot: AI writes code',
      desc: 'A GPT-3 descendant fine-tuned on public code became Copilot, an autocomplete that wrote whole functions from a comment. It was the first AI tool that millions of professionals used daily and paid for, and the first hint that programming would be the field where language models had their deepest impact.' },
    { t: 2021.55, year: '2021', tag: 'milestone', title: 'AlphaFold 2 solves protein structure prediction',
      desc: 'DeepMind\'s system predicted the three-dimensional shape of proteins from their amino-acid sequence to near-experimental accuracy, a fifty-year-old grand challenge of biology. Its attention-based architecture and the release of predicted structures for essentially every known protein earned Demis Hassabis and John Jumper the 2024 Nobel Prize in Chemistry.' },
    { t: 2022.07, year: '2022', tag: 'theory', title: 'InstructGPT: RLHF turns a text predictor into an assistant',
      desc: 'OpenAI fine-tuned GPT-3 on human demonstrations, trained a reward model on human preferences between outputs, and optimised the model against that reward. Human raters preferred a 1.3-billion-parameter InstructGPT to the 175-billion-parameter original. The paper is the recipe for ChatGPT, published ten months before ChatGPT (chapter 11 and lab 08).' },
    { t: 2022.22, year: '2022', tag: 'hardware', title: 'NVIDIA H100',
      desc: 'The Hopper GPU roughly tripled training throughput over the A100, added a transformer engine for 8-bit precision, and became the most sought-after object in the technology industry: export-controlled, allocated, and rented for several dollars an hour. Frontier runs of 2023–2025 were measured in "tens of thousands of H100s".' },
    { t: 2022.25, year: '2022', tag: 'theory', title: 'Chinchilla: most large models were undertrained', scale: 7e10, flop: 5.8e23,
      desc: 'DeepMind showed that for a fixed compute budget, parameters and training tokens should grow together, roughly 20 tokens per parameter. Their 70-billion-parameter Chinchilla, trained on 1.4 trillion tokens, beat the 280-billion-parameter Gopher. Labs stopped bragging about parameter counts and started counting tokens; later models were trained far past the Chinchilla point because inference cost matters too.' },
    { t: 2022.65, year: '2022', tag: 'product', title: 'Stable Diffusion: image generation, open and on your laptop',
      desc: 'Trained by CompVis at LMU Munich with Stability AI and Runway, and released with open weights, Stable Diffusion ran on a consumer GPU and spawned an ecosystem of fine-tunes, interfaces and controversies over training data and artists\' rights within weeks. It demonstrated how fast an open model spreads compared with an API.' },
    { t: 2022.92, year: '2022', tag: 'product', title: 'ChatGPT (30 November 2022)',
      desc: 'A "low-key research preview" of a chat interface on a GPT-3.5 model fine-tuned with RLHF reached a million users in five days and a hundred million in two months, the fastest-growing consumer application in history at the time. The underlying model was not new. The interface was. Every government, company and school on Earth suddenly had an AI policy to write.' },
    { t: 2023.15, year: '2023', tag: 'data', title: 'Llama: open weights at GPT-3 quality', scale: 6.5e10,
      desc: 'Meta released Llama (February, 7 to 65 billion parameters, trained on 1.4 trillion tokens) to researchers; the weights leaked within a week and were fine-tuned, quantised and run on laptops by hobbyists. Llama 2 followed in July with a licence allowing commercial use. The open-weights ecosystem (llama.cpp, Hugging Face, Mistral, Qwen, DeepSeek) dates from here.' },
    { t: 2023.2, year: '2023', tag: 'milestone', title: 'GPT-4 (14 March)', flop: 2e25, est: true,
      desc: 'OpenAI\'s multimodal model passed the bar exam in the top 10%, scored 5s on AP exams, and described images. The technical report disclosed no architecture, size or data, a first for a major model and a sign that the field had become an industry. Its training compute is estimated at around 2×10<sup>25</sup> FLOP, roughly a hundred times GPT-3.' },
    { t: 2023.22, year: '2023', tag: 'product', title: 'Claude (March)',
      desc: 'Anthropic, founded in 2021 by former OpenAI researchers with a safety-first mission, released its assistant, trained with Constitutional AI: the model critiques and revises its own outputs against a written set of principles, reducing the human labelling that RLHF needs. Claude 2 (July) brought a 100,000-token context window, at the time the longest available.' },
    { t: 2023.93, year: '2023', tag: 'product', title: 'Gemini 1.0 (December)', flop: 5e25, est: true,
      desc: 'Google DeepMind\'s first natively multimodal model family, trained on TPU v4 and v5 pods, was Google\'s answer to a year of being seen as behind despite having invented the transformer. Gemini Ultra was the first model to report beating human experts on the MMLU benchmark. The three-way race between OpenAI, Anthropic and Google was now explicit.' },
    { t: 2023.95, year: '2023', tag: 'architecture', title: 'Mixtral 8×7B: mixture-of-experts goes open', scale: 4.7e10,
      desc: 'Mistral AI, a Paris start-up less than a year old, released a model that routes each token to two of eight expert feed-forward blocks, so it matches a much larger dense model while using only 13 billion parameters per token. Sparse mixture-of-experts, an idea from the 1990s revived by Google in 2017–2021, became the standard frontier architecture.' },
    { t: 2024.12, year: '2024', tag: 'architecture', title: 'Gemini 1.5: a million-token context window',
      desc: 'Google shipped a model that could take an entire codebase, hours of video or several novels in a single prompt and answer questions about any part of it, with near-perfect recall in "needle in a haystack" tests. Long context changed what "retrieval" meant and made agents that read whole projects practical.' },
    { t: 2024.13, year: '2024', tag: 'product', title: 'Sora: a minute of coherent video from text',
      desc: 'OpenAI previewed a diffusion transformer that generated minute-long, physically plausible video clips, and described it as a step toward "world simulators". Publicly released in December 2024. Video generation, which in 2022 produced a few blurry seconds, had become a serious research direction on the road to models that understand the physical world.' },
    { t: 2024.18, year: '2024', tag: 'product', title: 'Claude 3 (March): Haiku, Sonnet, Opus',
      desc: 'Anthropic\'s three-size family, with Opus matching or beating GPT-4 on most benchmarks, made it a three-horse race in practice as well as in principle. Claude 3.5 Sonnet in June became the model of choice for coding; the "Artifacts" interface let it build and run small applications in the chat window.' },
    { t: 2024.22, year: '2024', tag: 'hardware', title: 'NVIDIA Blackwell',
      desc: 'Announced with 208 billion transistors across two dies, 4-bit precision support and rack-scale NVL72 systems that connect 72 GPUs as one, Blackwell was ordered by the hundreds of thousands before it shipped. NVIDIA briefly became the most valuable company in the world; GPUs had become the strategic resource of the decade.' },
    { t: 2024.56, year: '2024', tag: 'data', title: 'Llama 3.1 405B: open weights at the frontier, 15 trillion tokens', scale: 4.05e11, flop: 3.8e25,
      desc: 'Meta\'s Llama 3 arrived on 18 April 2024 at 8B and 70B; the 405-billion-parameter Llama 3.1 followed on 23 July. All of them were trained on 15 trillion tokens, about 40 times the Chinchilla-optimal amount for the 8B model, because a smaller model trained longer is cheaper to serve. The 405-billion-parameter version, trained on 16,000 H100s, was the first open-weights model at the frontier, and its 92-page report is the most detailed public account of how a frontier model is built.' },
    { t: 2024.7, year: '2024', tag: 'milestone', title: 'o1: reasoning models think before they answer',
      desc: 'OpenAI\'s o1 (September preview) was trained with reinforcement learning to produce a long hidden chain of thought before responding, and its accuracy on maths and coding scaled with how long it was allowed to think. Test-time compute became a second scaling axis alongside training compute. Within months every lab had a reasoning model.' },
    { t: 2024.9, year: '2024', tag: 'product', title: 'MCP: a standard way to plug tools into models',
      desc: 'Anthropic released the Model Context Protocol, an open standard for connecting assistants to data sources and tools, so that any tool written once works with any model. It was adopted across the industry during 2025 and became the plumbing of the agent era, the way HTTP was the plumbing of the web.' },
    { t: 2024.98, year: '2024', tag: 'milestone', title: 'DeepSeek-V3: frontier quality for a reported $5.6M of compute', scale: 6.71e11, flop: 3.3e24,
      desc: 'A Chinese lab spun out of a hedge fund released a 671-billion-parameter mixture-of-experts model (37 billion active) trained on 2.8 million H800 GPU-hours, with open weights and a candid technical report on multi-head latent attention, 8-bit training and load balancing. It matched models that reportedly cost ten times more to train and showed that engineering efficiency was a lever as powerful as raw compute.' },
    { t: 2025.05, year: '2025', tag: 'milestone', title: 'DeepSeek-R1: open reasoning, and a stock-market shock',
      desc: 'R1 reproduced o1-class reasoning with reinforcement learning on verifiable problems (the GRPO algorithm, rewards for correct answers and correct formatting) and published the recipe. Its release wiped hundreds of billions off NVIDIA\'s market value in a day on fears that compute mattered less than assumed. The fears faded; the recipe stayed and was copied everywhere.' },
    { t: 2025.25, year: '2025', tag: 'product', title: 'Agents and computer use go mainstream',
      desc: 'Models that browse, click, run code and work for hours without supervision moved from demos into products: coding agents that take a ticket and return a pull request, "deep research" agents that read hundreds of pages, and assistants that operate a computer through screenshots. The unit of AI work shifted from a single answer to a task, and the METR "task horizon" measurements in chapter 15 became the way progress was tracked.' },
    { t: 2025.39, year: '2025', tag: 'product', title: 'Claude 4 (May): Opus 4 and Sonnet 4',
      desc: 'Anthropic\'s fourth generation focused on long-running agentic coding, with models able to work autonomously for hours across large codebases, and shipped with the first activation of its ASL-3 safeguards under the Responsible Scaling Policy, a public sign that frontier labs were treating capability and risk as the same event.' },
    { t: 2025.6, year: '2025', tag: 'product', title: 'GPT-5 (August)',
      desc: 'OpenAI unified its fast models and its reasoning models behind a router that decides how long to think, and made the result the default for hundreds of millions of ChatGPT users. The reception was muted, which was itself news: the improvements were real but incremental, and the "is scaling slowing?" debate that fuels chapter 15 became mainstream.' },
    { t: 2025.74, year: '2025', tag: 'product', title: 'Claude Sonnet 4.5 (September)',
      desc: 'Positioned as the strongest coding and computer-use model of its generation, able to sustain focus on a single task for many hours, and the model behind the next wave of agent products. Mid-sized models were now doing what only the largest could a year earlier.' },
    { t: 2025.88, year: '2025', tag: 'product', title: 'Claude Opus 4.5 and Gemini 3 (November)',
      desc: 'Two frontier releases within a week of each other closed 2025. Anthropic\'s Opus 4.5 pushed agentic coding and long-horizon task reliability further; Google\'s Gemini 3 led on many reasoning and multimodal benchmarks. The pattern of the year: several labs at roughly the same frontier, leapfrogging every few months.' },
    { t: 2026.5, year: '2026', tag: 'product', title: 'The Claude 5 family, and a tier above Opus',
      desc: 'Anthropic\'s fifth generation arrived through 2026: Claude Fable 5 and Claude Mythos 5 in June — the same model at two levels of safeguard, sitting above the Opus class — then Sonnet 5 on 30 June, Opus 5 on 24 July, and Fable 5.1 and Mythos 5.1 on 1 September. Because this course is written in the middle of the year, this entry is deliberately brief: benchmark numbers date instantly and marketing claims are not history. Chapter 15 discusses what these models can and cannot do at the time of writing.' },
  ];
  /* extra points for the scale curve (not entries): today's estimated frontier */
  const EXTRA_POINTS = [
    { t: 2026.5, scale: 1.5e12, flop: 3e26, est: true, scaleEst: true, label: '2026 frontier (est.)' },
  ];

  function fmtBig(v) {
    if (v >= 1e12) return (v / 1e12).toFixed(v >= 1e13 ? 0 : 1) + ' trillion';
    if (v >= 1e9) return (v / 1e9).toFixed(v >= 1e10 ? 0 : 1) + ' billion';
    if (v >= 1e6) return (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + ' million';
    if (v >= 1e3) return (v / 1e3).toFixed(0) + ' thousand';
    return String(Math.round(v));
  }
  /* "4.7×10^17". Guards against 0 / negatives / non-finite values, and rolls 9.97×10^5 up to
     10^6 instead of printing the nonsense "10×10^5". */
  function fmtPow(v) {
    if (!isFinite(v) || v <= 0) return '—';
    let e = Math.floor(Math.log10(v));
    let m = v / Math.pow(10, e);
    if (m < 1) { m *= 10; e -= 1; }                 // floating-point rounding at exact powers
    if (m >= 9.95) { m = 1; e += 1; }
    return (m < 1.05 ? '10^' + e : m.toFixed(1) + '×10^' + e);
  }

  /* Frontier envelope of training compute, built from the entries above: only the runs that set a
     new record are kept. Both interactives and the worked example in the prose read from this. */
  const FLOP_ENV = (function () {
    const pts = ENTRIES.filter(e => e.flop).map(e => ({ t: e.t, v: e.flop, label: e.year }))
      .concat(EXTRA_POINTS.filter(p => p.flop).map(p => ({ t: p.t, v: p.flop, label: 'est.' })))
      .sort((a, b) => a.t - b.t);
    let run = 0; const out = [];
    for (const p of pts) if (p.v > run) { run = p.v; out.push({ t: p.t, l: Math.log10(p.v), label: p.label }); }
    return out;
  })();
  /* log10(FLOP) of the record run at a given year, interpolated; flat before the first record. */
  function logFlopAt(yr) {
    if (!FLOP_ENV.length) return 0;
    if (yr <= FLOP_ENV[0].t) return FLOP_ENV[0].l;
    for (let i = 1; i < FLOP_ENV.length; i++) {
      const a = FLOP_ENV[i - 1], b = FLOP_ENV[i], dt = b.t - a.t;
      if (yr <= b.t) return dt > 0 ? a.l + (b.l - a.l) * (yr - a.t) / dt : b.l;
    }
    return FLOP_ENV[FLOP_ENV.length - 1].l;
  }
  /* Doubling time in months implied by the records from `from` onwards (null if not computable). */
  function dataDoublingMonths(from) {
    const pts = FLOP_ENV.filter(p => p.t >= from - 0.5);
    if (pts.length < 2) return null;
    const a = pts[0], b = pts[pts.length - 1], dt = b.t - a.t, dl = b.l - a.l;
    if (!(dt > 0) || !(dl > 0)) return null;
    return 12 * Math.log10(2) * dt / dl;
  }

  /* =====================================================================
     Interactive (a): the timeline canvas
     ===================================================================== */
  function buildTimeline(ctx) {
    const h = ctx.h;
    const W = 720, HH = 440;
    const [cv, g] = ctx.canvas(W, HH);
    cv.style.touchAction = 'none'; cv.style.cursor = 'grab';
    const padL = 48, padR = 16, axisY = 350, laneY0 = 326, laneGap = 12, maxLanes = 15;
    const TITLE_FONT = '10px Inter, sans-serif';
    const curveTop = 22, curveBot = 132, histTop = 376, histBot = 420;
    const MINY = 1938, MAXY = 2030, BUCKET = 5;
    const FIT0 = 1940, FIT1 = 2028;
    let t0 = FIT0, t1 = FIT1;
    let hover = null, selected = null, curveMode = 'params', placed = [], dirty = true;
    let cardEntry = 0;                       // 0 = nothing rendered yet (null is a real state)
    const enabled = {}; TAGS.forEach(t => { enabled[t.id] = true; });
    const pointers = new Map(); let drag = null, pinch = null, moved = false;
    const mark = () => { dirty = true; };
    const buckets = [];
    for (let y = FIT0; y < MAXY; y += BUCKET) buckets.push({ a: y, n: 0 });

    const card = h('div', { style: { marginTop: '10px', minHeight: '120px', padding: '12px 14px', background: '#111827', border: '1px solid #243044', borderRadius: '10px', fontSize: '.92rem' } });
    const pxPerYear = () => (W - padL - padR) / (t1 - t0);
    const xOf = (yr) => padL + (yr - t0) * pxPerYear();
    const yearOf = (x) => t0 + (x - padL) / pxPerYear();

    function setView(a, b) {
      let span = b - a;
      if (!isFinite(a) || !isFinite(span)) { a = FIT0; span = FIT1 - FIT0; }
      span = Math.max(1.5, Math.min(MAXY - MINY, span));
      if (a < MINY) a = MINY;
      if (a + span > MAXY) a = MAXY - span;
      if (a !== t0 || a + span !== t1) mark();
      t0 = a; t1 = a + span;
    }
    function visible() { return ENTRIES.filter(e => enabled[e.tag]); }
    /* Lanes are packed on the full width each entry will occupy — the dot AND the
       title printed beside it once the view is zoomed in far enough to show one.
       Packing on the dot alone put four 150px titles in one 12px lane. */
    function layout() {
      placed = [];
      for (const b of buckets) b.n = 0;
      const R = pxPerYear() > 18 ? 5 : 4;
      const named = pxPerYear() > 55;
      if (named) g.font = TITLE_FONT;
      for (const e of visible()) {
        const bi = Math.floor((e.t - FIT0) / BUCKET);
        if (bi >= 0 && bi < buckets.length) buckets[bi].n++;
        const x = xOf(e.t);
        if (x < padL - 20 || x > W - padR + 20) continue;
        let label = null, x0 = x - R - 1, x1 = x + R + 1;
        if (named) {
          const txt = e.title.length > 34 ? e.title.slice(0, 33) + '…' : e.title;
          const w = g.measureText(txt).width;
          const right = x + 9 + w < W - padR;
          label = { txt: txt, right: right };
          if (right) x1 = x + 9 + w + 5; else x0 = x - 9 - w - 5;
        }
        let lane = 0;
        while (lane < maxLanes - 1 && placed.some(p => p.lane === lane && p.x1 > x0 && p.x0 < x1)) lane++;
        placed.push({ e, x, y: laneY0 - lane * laneGap, lane, r: R, label: label, x0: x0, x1: x1 });
      }
    }
    /* Rebuilds the card only when the entry actually changes (pointermove fires constantly). */
    function showCard(e) {
      if (e === cardEntry) return;
      cardEntry = e;
      card.innerHTML = '';
      if (!e) { card.append(h('div', { class: 'muted' }, 'Hover or tap a dot to read its story. Drag to pan, scroll or pinch to zoom, use the filters to isolate a thread (try "hardware" alone, or "winter").')); return; }
      const bits = [];
      if (e.scale) bits.push('scale: ' + fmtBig(e.scale) + ' parameters (' + fmtPow(e.scale) + ')');
      if (e.flop) bits.push('training compute: ~' + fmtPow(e.flop) + ' FLOP' + (e.est ? ' (estimate)' : ''));
      card.append(
        h('div', { style: { display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '4px' } },
          h('span', { style: { fontFamily: 'JetBrains Mono, monospace', color: '#7c9cff', fontSize: '.85rem' } }, e.year),
          h('span', { class: 'pill', style: { color: TAGCOL[e.tag], borderColor: TAGCOL[e.tag] } }, e.tag)),
        h('div', { style: { fontWeight: 600, fontSize: '1.02rem', marginBottom: '6px' } }, e.title),
        h('div', { class: 'muted', html: e.desc }),
        bits.length ? h('div', { style: { marginTop: '8px', fontFamily: 'JetBrains Mono, monospace', fontSize: '.78rem', color: '#38d9a9' } }, bits.join(' · ')) : null);
    }

    function draw() {
      g.clearRect(0, 0, W, HH); g.fillStyle = ctx.colors.bg; g.fillRect(0, 0, W, HH);
      layout();
      // ---- winters shading
      for (const e of ENTRIES) if (e.tag === 'winter' && enabled.winter && e.end) {
        const x0 = Math.max(padL, xOf(e.t)), x1 = Math.min(W - padR, xOf(e.end));
        if (x1 > x0) { g.fillStyle = 'rgba(251,113,133,0.07)'; g.fillRect(x0, curveBot + 10, x1 - x0, axisY - curveBot - 10); }
      }
      // ---- scale curve band
      const isP = curveMode === 'params';
      const vKey = isP ? 'scale' : 'flop', eKey = isP ? 'scaleEst' : 'est';
      const lo = isP ? 2 : 16, hi = isP ? 13 : 28, gridStep = isP ? 2 : 3;
      const yOfLog = (l) => curveBot - (ctx.clamp(l, lo, hi) - lo) / (hi - lo) * (curveBot - curveTop);
      g.strokeStyle = ctx.colors.line; g.lineWidth = 1; g.font = '10px JetBrains Mono, monospace'; g.fillStyle = ctx.colors.muted; g.textAlign = 'right';
      for (let l = lo; l <= hi; l += gridStep) { const y = yOfLog(l); g.beginPath(); g.moveTo(padL, y); g.lineTo(W - padR, y); g.stroke(); g.fillText('10^' + l, padL - 5, y + 3); }
      const pts = ENTRIES.filter(e => e[vKey] > 0).map(e => ({ t: e.t, v: e[vKey], est: !!e[eKey] }))
        .concat(EXTRA_POINTS.filter(p => p[vKey] > 0).map(p => ({ t: p.t, v: p[vKey], est: !!p[eKey] })))
        .sort((a, b) => a.t - b.t);
      let run = 0; const env = [];
      for (const p of pts) { run = Math.max(run, p.v); env.push({ t: p.t, v: run, est: p.est, raw: p.v }); }
      /* the record curve is clipped to the band: when the view is zoomed into a
         late decade it enters from far off to the left, and unclipped it ran
         straight across the 10^N labels in the margin */
      g.save(); g.beginPath(); g.rect(padL, curveTop - 10, W - padL - padR, curveBot - curveTop + 20); g.clip();
      g.strokeStyle = ctx.colors.green; g.lineWidth = 2; g.beginPath();
      env.forEach((p, i) => { const x = xOf(p.t), y = yOfLog(Math.log10(p.v)); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); });
      if (env.length) g.stroke();
      for (const p of env) {
        const x = xOf(p.t), y = yOfLog(Math.log10(p.raw));
        /* one marker half-width inside the band, so the clip never bisects a square */
        if (x < padL + 5 || x > W - padR - 5) continue;
        g.fillStyle = p.est ? ctx.colors.bg : ctx.colors.green; g.strokeStyle = ctx.colors.green; g.lineWidth = 1.5;
        g.beginPath(); g.rect(x - 3, y - 3, 6, 6); g.fill(); g.stroke();
      }
      g.restore();
      g.fillStyle = ctx.colors.green; g.textAlign = 'left'; g.font = '11px Inter, sans-serif';
      g.fillText(isP ? 'largest model in this timeline: parameters (log scale; line = record so far, hollow = estimate)'
        : 'training compute per run, FLOP (log scale; line = record so far, hollow = estimate)', padL + 6, curveTop - 6);
      // ---- axis
      g.strokeStyle = ctx.colors.muted; g.lineWidth = 1; g.beginPath(); g.moveTo(padL, axisY); g.lineTo(W - padR, axisY); g.stroke();
      const span = t1 - t0, stepY = span > 60 ? 10 : span > 25 ? 5 : span > 8 ? 2 : 1;
      g.font = '11px JetBrains Mono, monospace'; g.textAlign = 'center';
      for (let yr = Math.ceil(t0 / stepY) * stepY; yr <= t1; yr += stepY) {
        const x = xOf(yr); g.strokeStyle = ctx.colors.line; g.beginPath(); g.moveTo(x, curveBot + 8); g.lineTo(x, axisY + 5); g.stroke();
        g.fillStyle = ctx.colors.muted; g.fillText(String(Math.round(yr)), x, axisY + 18);
      }
      // ---- stems, then dots (two passes so no stem crosses a dot)
      g.strokeStyle = 'rgba(148,163,184,0.25)'; g.lineWidth = 1;
      for (const p of placed) { g.beginPath(); g.moveTo(p.x, p.y + p.r); g.lineTo(p.x, axisY); g.stroke(); }
      for (const p of placed) {
        const isHov = hover && hover.e === p.e, isSel = selected === p.e;
        g.fillStyle = TAGCOL[p.e.tag]; g.beginPath(); g.arc(p.x, p.y, isHov || isSel ? p.r + 2 : p.r, 0, Math.PI * 2); g.fill();
        if (isSel) { g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath(); g.arc(p.x, p.y, p.r + 5, 0, Math.PI * 2); g.stroke(); }
      }
      // ---- titles beside the dots once there is room (the lane packing above
      //      already reserved the width each one needs)
      g.font = TITLE_FONT;
      for (const p of placed) {
        if (!p.label) continue;
        g.textAlign = p.label.right ? 'left' : 'right';
        g.fillStyle = TAGCOL[p.e.tag]; g.fillText(p.label.txt, p.x + (p.label.right ? 9 : -9), p.y + 3);
      }
      // ---- tooltip for the hovered / selected entry
      const focus = hover || (selected ? placed.find(p => p.e === selected) : null);
      if (focus) {
        g.font = '600 12px Inter, sans-serif'; const txt = focus.e.year + ' · ' + focus.e.title;
        const tw = Math.min(W - padL - padR, g.measureText(txt).width + 14);
        let bx = focus.x + 12; if (bx + tw > W - padR) bx = focus.x - 12 - tw;
        bx = ctx.clamp(bx, padL, W - padR - tw);
        let by = focus.y - 26; if (by < curveBot + 12) by = focus.y + 12;
        g.fillStyle = '#172033'; g.strokeStyle = TAGCOL[focus.e.tag]; g.lineWidth = 1; g.beginPath(); g.rect(bx, by, tw, 20); g.fill(); g.stroke();
        g.fillStyle = ctx.colors.text; g.textAlign = 'left'; g.fillText(txt, bx + 7, by + 14);
      }
      // ---- density strip: entries per 5 years, the acceleration in one picture
      let maxN = 1; for (const b of buckets) if (b.n > maxN) maxN = b.n;
      for (const b of buckets) {
        if (!b.n) continue;
        const x0 = Math.max(padL, xOf(b.a)), x1 = Math.min(W - padR, xOf(b.a + BUCKET));
        if (x1 - x0 < 1.5) continue;
        const bh = (histBot - histTop) * b.n / maxN;
        g.fillStyle = 'rgba(124,156,255,0.22)'; g.fillRect(x0, histBot - bh, Math.max(1, x1 - x0 - 1), bh);
        if (x1 - x0 > 24) {
          /* the count sits above its bar, but the tallest bar reaches the top of
             the strip, so that one is labelled inside itself instead of up in the
             row of year labels */
          g.fillStyle = ctx.colors.accent; g.font = '10px JetBrains Mono, monospace'; g.textAlign = 'center';
          g.fillText(String(b.n), (x0 + x1) / 2, bh > 16 ? histBot - bh + 11 : histBot - bh - 3);
        }
      }
      /* the strip's own caption goes on the bottom line: at histTop it landed in
         the middle of the year labels */
      g.fillStyle = ctx.colors.muted; g.font = '10px Inter, sans-serif'; g.textAlign = 'left';
      g.fillText('entries per 5 years', padL, HH - 6);
      g.textAlign = 'right';
      g.fillText(Math.round(t0) + ' – ' + Math.round(t1) + '  ·  drag to pan · wheel / pinch to zoom', W - padR, HH - 6);
    }

    function nearest(x, y) {
      let best = null, bd = 169;               // within 13 px of a dot centre
      for (const p of placed) { const d = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y); if (d < bd) { bd = d; best = p; } }
      return best;
    }
    cv.addEventListener('pointerdown', (ev) => {
      try { cv.setPointerCapture(ev.pointerId); } catch (e) {}
      const p = cv.pos(ev); pointers.set(ev.pointerId, p); moved = false;
      if (pointers.size === 1) { drag = { x: p.x, t0, t1 }; pinch = null; cv.style.cursor = 'grabbing'; }
      else if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = { d: Math.max(1, Math.abs(a.x - b.x)), mid: (a.x + b.x) / 2, t0, t1 }; drag = null; }
      if (ev.preventDefault) ev.preventDefault();
    });
    cv.addEventListener('pointermove', (ev) => {
      const p = cv.pos(ev);
      if (pointers.has(ev.pointerId)) pointers.set(ev.pointerId, p);
      if (pinch && pointers.size >= 2) {
        const [a, b] = [...pointers.values()]; const d = Math.max(1, Math.abs(a.x - b.x)), mid = (a.x + b.x) / 2;
        const span0 = pinch.t1 - pinch.t0, span = ctx.clamp(span0 * pinch.d / d, 1.5, MAXY - MINY);
        const yearAtMid = pinch.t0 + (pinch.mid - padL) / (W - padL - padR) * span0;
        const a0 = yearAtMid - (mid - padL) / (W - padL - padR) * span;
        setView(a0, a0 + span); moved = true;
      } else if (drag) {
        const dx = p.x - drag.x; if (Math.abs(dx) > 3) moved = true;
        const shift = dx / ((W - padL - padR) / (drag.t1 - drag.t0));
        setView(drag.t0 - shift, drag.t1 - shift);
      } else {
        const n = nearest(p.x, p.y);
        if (n !== hover) { hover = n; mark(); }
        cv.style.cursor = n ? 'pointer' : 'grab';
        if (!selected) showCard(n ? n.e : null);
      }
    });
    const up = (ev) => {
      const p = cv.pos(ev);
      if (!moved && drag) {
        const n = nearest(p.x, p.y);
        selected = (n && selected !== n.e) ? n.e : null;
        showCard(selected || (n ? n.e : null)); mark();
      }
      pointers.delete(ev.pointerId);
      try { cv.releasePointerCapture(ev.pointerId); } catch (e) {}
      if (pointers.size === 1) { drag = { x: [...pointers.values()][0].x, t0, t1 }; pinch = null; }  // second finger lifted: keep panning
      else { drag = null; pinch = null; }
      cv.style.cursor = 'grab';
    };
    cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    cv.addEventListener('pointerleave', () => {
      if (!drag && !pinch) { if (hover) { hover = null; mark(); } if (!selected) showCard(null); }
    });
    cv.addEventListener('wheel', (ev) => {
      if (ev.preventDefault) ev.preventDefault();
      const p = cv.pos(ev); const f = Math.exp(ctx.clamp(ev.deltaY || 0, -300, 300) * 0.0015);
      const yr = yearOf(p.x); const span = (t1 - t0) * f;
      const a = yr - (p.x - padL) / (W - padL - padR) * span;
      setView(a, a + span);
    }, { passive: false });

    ctx.loop(() => { if (dirty) { dirty = false; draw(); } });
    showCard(null);

    const boxes = TAGS.map(t => {
      const cb = h('input', { type: 'checkbox', checked: true, style: { accentColor: t.color } });
      cb.addEventListener('change', () => {
        enabled[t.id] = !!cb.checked;
        if (selected && !enabled[selected.tag]) { selected = null; showCard(null); }
        mark();
      });
      cb._tag = t.id;
      return cb;
    });
    const filterRow = h('div', { style: { display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center' } },
      h('span', { class: 'muted', style: { fontSize: '.8rem' } }, 'show:'),
      ...boxes.map((cb, i) => h('label', { style: { display: 'flex', alignItems: 'center', gap: '6px', fontSize: '.82rem', color: TAGS[i].color, cursor: 'pointer' } }, cb, TAGS[i].label)));
    const stepSel = (dir) => {
      const vis = visible(); if (!vis.length) return;
      let i = selected ? vis.indexOf(selected) : (dir > 0 ? -1 : 0);
      i = ((i + dir) % vis.length + vis.length) % vis.length;
      selected = vis[i]; showCard(selected); mark();
      if (selected.t < t0 + 1 || selected.t > t1 - 1) { const sp = t1 - t0; setView(selected.t - sp / 2, selected.t + sp / 2); }
    };
    const btns = h('div', { class: 'btn-row', style: { margin: 0 } },
      ctx.button('◀ prev', () => stepSel(-1)), ctx.button('next ▶', () => stepSel(1)),
      ctx.button('fit all', () => setView(FIT0, FIT1)),
      ctx.button('1940–1990', () => setView(1940, 1990)),
      ctx.button('2010–2027', () => setView(2010, 2027)),
      ctx.button('2022–2027', () => setView(2022, 2027)),
      ctx.button('↺ reset', () => {
        boxes.forEach(cb => { cb.checked = true; enabled[cb._tag] = true; });
        selected = null; hover = null; showCard(null); setView(FIT0, FIT1); mark();
      }));
    const curveSel = ctx.select({
      label: 'curve above the axis',
      options: [{ value: 'params', label: 'parameters of the largest model' }, { value: 'flop', label: 'training compute (FLOP)' }],
      value: 'params', onChange: v => { curveMode = v; mark(); },
    });
    const body = h('div', {}, cv, card);
    return ctx.figure(body, 'Every entry in this chapter as a dot, coloured by kind; entries from the same year fan upward. The green line above the axis is the record so far — the largest parameter count (or training compute) reached by any entry up to that date — on a log scale, so each gridline is a hundred-fold (params) or thousand-fold (FLOP) jump. Squares are individual models, so a square below the line is a model that was deliberately smaller than the record, like Chinchilla. Hollow squares are estimates: labs stopped publishing sizes after GPT-3. The blue bars under the axis count entries per five years, and they are the whole argument of this chapter in one picture.', [btns, filterRow, curveSel]);
  }

  /* =====================================================================
     Interactive (b): compute doubling explorer
     ===================================================================== */
  function buildDoubling(ctx) {
    let months = 6, start = 2012;
    const NOW = 2026.7, X0 = 2012, XMAX = 2030, LMAX = 16;
    const [cv, g] = ctx.canvas(720, 300);
    const rd = ctx.readout();
    const refs = [
      { m: 3.4, col: ctx.colors.warn, label: '3.4-month doubling — OpenAI, "AI and Compute" (2012–2018)' },
      { m: 6, col: ctx.colors.green, label: '6-month doubling — Epoch AI, ~4–5× per year' },
      { m: 24, col: ctx.colors.muted, label: '24-month doubling — Moore\'s law, for comparison' },
    ];
    function draw() {
      const W = cv.W, HH = cv.H;
      g.clearRect(0, 0, W, HH); g.fillStyle = ctx.colors.bg; g.fillRect(0, 0, W, HH);
      const padL = 54, padR = 16, padT = 16, padB = 28, pw = W - padL - padR, ph = HH - padT - padB;
      const xOf = yr => padL + (ctx.clamp(yr, X0, XMAX) - X0) / (XMAX - X0) * pw;
      const yOf = l => padT + ph * (1 - ctx.clamp(l, 0, LMAX) / LMAX);
      // grid
      g.strokeStyle = ctx.colors.line; g.lineWidth = 1; g.font = '11px JetBrains Mono, monospace'; g.fillStyle = ctx.colors.muted;
      for (let l = 0; l <= LMAX; l += 2) { const y = yOf(l); g.beginPath(); g.moveTo(padL, y); g.lineTo(W - padR, y); g.stroke(); g.textAlign = 'right'; g.fillStyle = ctx.colors.muted; g.fillText('10^' + l + '×', padL - 6, y + 4); }
      for (let yr = X0; yr <= XMAX; yr += 2) { const x = xOf(yr); g.strokeStyle = ctx.colors.line; g.beginPath(); g.moveTo(x, padT); g.lineTo(x, padT + ph); g.stroke(); g.textAlign = 'center'; g.fillStyle = ctx.colors.muted; g.fillText(String(yr), x, HH - 8); }
      // the four exponentials
      const curves = refs.concat([{ m: months, col: ctx.colors.accent, label: 'your assumption: ' + months.toFixed(1) + '-month doubling', main: true }]);
      for (const c of curves) {
        const m = Math.max(0.25, c.m);
        g.strokeStyle = c.col; g.lineWidth = c.main ? 3 : 1.5; g.setLineDash(c.main ? [] : [5, 4]); g.beginPath();
        let first = true;
        for (let yr = start; yr <= XMAX + 1e-9; yr += 0.1) {
          const l = ((yr - start) * 12 / m) * Math.log10(2);   // log10 of 2^(elapsed months / m)
          if (l > LMAX) break;
          const x = xOf(yr), y = yOf(l); if (first) { g.moveTo(x, y); first = false; } else g.lineTo(x, y);
        }
        g.stroke();
      }
      g.setLineDash([]);
      // what actually happened: the record runs from the timeline, relative to the start year
      const base = logFlopAt(start);
      const shown = FLOP_ENV.filter(p => p.t >= start - 0.5 && p.t <= XMAX);
      g.fillStyle = ctx.colors.pink; g.strokeStyle = ctx.colors.pink; g.lineWidth = 1.5; g.beginPath();
      shown.forEach((p, i) => { const x = xOf(p.t), y = yOf(Math.max(0, p.l - base)); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); });
      if (shown.length > 1) g.stroke();
      for (const p of shown) {
        const x = xOf(p.t), y = yOf(Math.max(0, p.l - base));
        g.fillStyle = ctx.colors.pink; g.beginPath(); g.arc(x, y, 3.5, 0, Math.PI * 2); g.fill();
      }
      // today
      const xn = xOf(NOW); g.strokeStyle = ctx.colors.danger; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(xn, padT); g.lineTo(xn, padT + ph); g.stroke(); g.setLineDash([]);
      g.fillStyle = ctx.colors.danger; g.textAlign = 'left'; g.font = '11px Inter, sans-serif'; g.fillText('today', xn + 4, padT + ph - 6);
      // legend, on its own panel so it stays readable over the gridlines
      const lines = curves.concat([{ col: ctx.colors.pink, label: 'what actually happened (record runs in this chapter)' }]);
      g.fillStyle = 'rgba(10,14,22,0.85)'; g.fillRect(padL + 4, padT + 4, 368, lines.length * 15 + 8);
      g.font = '11px Inter, sans-serif'; let ly = padT + 19;
      for (const c of lines) { g.fillStyle = c.col; g.textAlign = 'left'; g.fillText((c.main ? '■ ' : c.m ? '- - ' : '● ') + c.label, padL + 10, ly); ly += 15; }
      // numbers
      const yrs = Math.max(0, NOW - start), dbl = yrs * 12 / Math.max(0.25, months), mult = Math.pow(2, dbl);
      const obs = dataDoublingMonths(start);
      rd.set({
        'from': start.toFixed(0) + ' to today', 'elapsed': yrs.toFixed(1) + ' yr',
        'your doublings': dbl.toFixed(1), 'your multiplier': fmtPow(mult),
        'at 3.4 mo': fmtPow(Math.pow(2, yrs * 12 / 3.4)), 'at 6 mo': fmtPow(Math.pow(2, yrs * 12 / 6)), 'at 24 mo': fmtPow(Math.pow(2, yrs * 12 / 24)),
        'records imply': obs ? obs.toFixed(1) + ' months' : 'n/a',
      });
    }
    const s1 = ctx.slider({ label: 'doubling time (months)', min: 1, max: 24, step: 0.1, value: months, digits: 1, onChange: v => { months = v; draw(); } });
    const s2 = ctx.slider({ label: 'start year', min: 2012, max: 2024, step: 1, value: start, onChange: v => { start = v; draw(); } });
    const reset = ctx.button('↺ reset', () => { months = 6; start = 2012; s1.value = 6; s2.value = 2012; draw(); });
    draw();
    return ctx.figure(cv, 'How much more compute the largest training run uses, relative to the start year, under different doubling times. The y-axis is logarithmic: every gridline is a hundred-fold. The pink line is not a model — it is this chapter\'s own record runs (AlexNet, the Transformer, GPT-2, GPT-3, Chinchilla, GPT-4, Gemini, and a 2026 estimate), plotted the same way, so you can slide the blue line until it lies on top of them. The references are approximate: OpenAI\'s 2018 analysis found a 3.4-month doubling from AlexNet to AlphaGo Zero, and Epoch AI estimates frontier training compute has grown roughly 4–5× per year since 2010, a doubling every six months or so.', [s1, s2, reset], rd);
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
  /* Interactive: how badly experts have called the timing               */
  /* ------------------------------------------------------------------ */
  function buildPredictionGame(ctx) {
    const [cv, g] = ctx.canvas(720, 340);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    const Q = [
      {
        claim: '"Within ten years a digital computer will be the world\'s chess champion."',
        who: 'Herbert Simon and Allen Newell, 1957',
        predicted: 1967, actual: 1997,
        note: 'Deep Blue beat Kasparov forty years after the prediction, and it did it by searching 200 million positions a second, not by thinking like a person.',
      },
      {
        claim: '"Machines will be capable of doing any work a man can do."',
        who: 'Herbert Simon, 1965, giving it twenty years',
        predicted: 1985, actual: null,
        note: 'Still not true in 2026. This prediction helped inflate the expectations whose collapse caused the first AI winter.',
      },
      {
        claim: 'The perceptron "will be able to walk, talk, see, write, reproduce itself and be conscious of its existence."',
        who: 'The New York Times, reporting the US Navy, July 1958',
        predicted: 1970, actual: null,
        note: 'Rosenblatt\'s machine could draw one straight line. Chapter 1 had you fail at XOR with it. The gap between the press release and the device is the oldest story in this field.',
      },
      {
        claim: '"Solving Go is at least ten years away."',
        who: 'The consensus view among Go and AI researchers, 2014',
        predicted: 2024, actual: 2016,
        note: 'This one went the other way. AlphaGo beat Lee Sedol two years later — the field has been badly wrong in both directions.',
      },
      {
        claim: '"A self-driving car will be commercially available by 2020."',
        who: 'Widely repeated industry guidance, around 2015',
        predicted: 2020, actual: null,
        note: 'Limited robotaxi services run in a handful of mapped cities. General self-driving remains unsolved, more than a decade after it was called nearly done.',
      },
    ];
    let idx = 0, guess = 2000, revealed = false;
    const gSl = ctx.slider({ label: 'your guess: when did it actually happen?', min: 1955, max: 2040, step: 1, value: 2000, fmt: (v) => String(Math.round(v)), onChange: (v) => { guess = v; } });
    const revBtn = ctx.button('Reveal', () => { revealed = true; }, 'primary');
    const nextBtn = ctx.button('Next prediction →', () => { idx = (idx + 1) % Q.length; revealed = false; });
    const ro = ctx.readout();

    ctx.loop(() => {
      g.clearRect(0, 0, cv.W, cv.H);
      const q = Q[idx];
      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('a confident prediction', 34, 26);
      g.fillStyle = 'rgba(124,156,255,0.10)'; g.fillRect(34, 38, 650, 74);
      g.strokeStyle = C.accent; g.lineWidth = 1.5; g.strokeRect(34, 38, 650, 74);
      g.font = '14px Inter, system-ui, sans-serif'; g.fillStyle = C.text;
      wrapLines(g, q.claim, 610).slice(0, 3).forEach((ln, i) => g.fillText(ln, 48, 62 + i * 20));
      g.font = MONO; g.fillStyle = C.muted;
      g.fillText('— ' + q.who, 48, 104);

      /* the timeline strip */
      const X = 60, W = 600, Y = 160;
      const TICKS = [1960, 1980, 2000, 2020, 2040];
      const tx = (yr) => X + (yr - 1955) / (2040 - 1955) * W;
      /* an opaque chip cut out of the background, so a label on the strip is
         read against the page rather than against whatever line runs under it */
      const chip = (s, cx, base) => {
        const w = g.measureText(s).width;
        g.fillStyle = C.bg; g.fillRect(cx - w / 2 - 5, base - 13, w + 10, 18);
      };
      g.strokeStyle = C.line; g.lineWidth = 2;
      g.beginPath(); g.moveTo(X, Y); g.lineTo(X + W, Y); g.stroke();
      TICKS.forEach(yr => { g.beginPath(); g.moveTo(tx(yr), Y - 5); g.lineTo(tx(yr), Y + 5); g.stroke(); });
      /* Every rule on the strip is drawn first and every label afterwards, so
         the full-height guess line passes behind the year labels it crosses
         instead of striking them out. */
      g.font = MONO; g.textAlign = 'center';
      if (!revealed) {
        g.strokeStyle = C.accent; g.lineWidth = 2.5;
        /* stops at the top of the year-label chips: a 3px stub poking out below
           them reads as a broken line, not as a marker */
        g.beginPath(); g.moveTo(tx(guess), Y - 30); g.lineTo(tx(guess), Y + 9); g.stroke();
      } else if (q.actual) {
        g.strokeStyle = 'rgba(251,191,36,0.6)'; g.lineWidth = 2; g.setLineDash([4, 4]);
        g.beginPath(); g.moveTo(tx(q.predicted), Y - 8); g.lineTo(tx(q.actual), Y - 8); g.stroke(); g.setLineDash([]);
        g.fillStyle = C.green;
        g.beginPath(); g.arc(tx(q.actual), Y, 8, 0, 7); g.fill();
      }
      /* predicted */
      g.fillStyle = C.warn;
      g.beginPath(); g.moveTo(tx(q.predicted), Y - 8); g.lineTo(tx(q.predicted) + 6, Y - 20); g.lineTo(tx(q.predicted) - 6, Y - 20); g.closePath(); g.fill();
      chip('promised', tx(q.predicted), Y - 26);
      g.fillStyle = C.warn; g.fillText('promised', tx(q.predicted), Y - 26);
      /* the year labels, last of all */
      TICKS.forEach(yr => {
        chip(String(yr), tx(yr), Y + 22);
        g.fillStyle = C.muted; g.fillText(String(yr), tx(yr), Y + 22);
      });
      /* and the reader's own marker, named on the row below the years */
      if (!revealed) {
        g.fillStyle = C.accent; g.fillText('you: ' + Math.round(guess), tx(guess), Y + 46);
      } else if (q.actual) {
        g.font = 'bold ' + MONO; g.fillStyle = C.green;
        g.fillText('actually ' + q.actual, tx(q.actual), Y + 44);
      } else {
        g.font = 'bold ' + MONO; g.fillStyle = C.danger;
        g.fillText('still has not happened', tx(2028), Y + 44);
      }
      g.textAlign = 'left';

      if (revealed) {
        const err = q.actual ? q.actual - q.predicted : null;
        g.font = 'bold 16px Inter, system-ui, sans-serif';
        g.fillStyle = err === null ? C.danger : C.warn;
        g.fillText(err === null
          ? 'Never. Sixty years of "twenty years away".'
          : (err > 0 ? 'Late by ' + err + ' years.' : 'Early by ' + (-err) + ' years — the field got this one badly wrong in the other direction.'),
          34, 244);
        g.font = FONT; g.fillStyle = C.muted;
        wrapText(g, q.note, 34, 268, 640, 17);
      } else {
        g.font = FONT; g.fillStyle = C.muted;
        wrapText(g, 'Drag the slider to when you think it actually happened, then press Reveal. The point is not to catch anyone out — it is that the people making these predictions were the leading experts of their day, with full knowledge of the state of the art.', 34, 250, 640, 17);
      }
      ro.set({ prediction: idx + 1 + ' / ' + Q.length, promised: q.predicted, actual: revealed ? (q.actual || 'still waiting') : '—' });
    });

    return ctx.figure(cv,
      'Every claim here was made by someone with full command of the state of the art. Two were decades late, one was eight years early, and two have not happened at all. This is the single most useful thing history offers about the present: the field has a very long record of being confidently wrong about timing in both directions, which is worth remembering next time you read a forecast — including the ones in chapter 15.',
      [gSl, revBtn, nextBtn], ro);
  }

  /* ------------------------------------------------------------------ */
  /* Interactive: what was actually possible in a given year             */
  /* ------------------------------------------------------------------ */
  function buildStateOfArt(ctx) {
    const [cv, g] = ctx.canvas(720, 376);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    let year = 2012;
    const ROWS = [
      {
        n: 'recognise objects in a photo',
        at: [[1998, 'only hand-written digits, on cheques'], [2012, 'AlexNet: 1,000 categories, 15% top-5 error'], [2015, 'ResNet passes human accuracy on ImageNet'], [2021, 'zero-shot: name a category it was never trained on']],
      },
      {
        n: 'translate a sentence',
        at: [[1998, 'phrase tables and hand-written rules; word salad'], [2014, 'seq2seq LSTMs: fluent but forgetful on long sentences'], [2016, 'Google Translate switches to neural; errors drop ~60%'], [2022, 'near-professional on high-resource language pairs']],
      },
      {
        n: 'hold a conversation',
        at: [[1998, 'ELIZA-style pattern matching, and everyone knew'], [2014, 'scripted assistants with a fixed intent list'], [2020, 'GPT-3: fluent, but you had to coax it with examples'], [2022, 'ChatGPT: ask a question, get an answer']],
      },
      {
        n: 'write working code',
        at: [[1998, 'autocomplete of variable names'], [2014, 'snippet search'], [2021, 'Copilot completes a function from its name'], [2025, 'agents that edit a repo, run the tests, and fix what fails']],
      },
      {
        n: 'play Go at professional level',
        at: [[1998, 'strong amateurs beat the best programs easily'], [2014, 'still considered a decade away'], [2016, 'AlphaGo beats Lee Sedol 4-1'], [2017, 'AlphaGo Zero learns it from self-play alone, 100-0']],
      },
    ];
    const YEARS = [1998, 2012, 2014, 2016, 2020, 2022, 2025];
    const ySl = ctx.slider({ label: 'year', min: 1998, max: 2026, step: 1, value: 2012, fmt: (v) => String(Math.round(v)), onChange: (v) => { year = v; } });
    const presets = [1998, 2014, 2020, 2025].map(y => ctx.button(String(y), () => { year = y; ySl.value = y; }));
    const ro = ctx.readout();

    /* three columns: what you want to do | the best that could do it | since when.
       The middle column stops at MIDW so it cannot run into the date on the right. */
    const MIDX = 300, MIDW = 296, SINCEX = 686;
    ctx.loop(() => {
      g.clearRect(0, 0, cv.W, cv.H);
      g.font = 'bold 17px Inter, system-ui, sans-serif'; g.fillStyle = C.text;
      g.fillText('the best anyone could do in ' + Math.round(year), 34, 30);
      let solved = 0;
      ROWS.forEach((r, i) => {
        const y = 58 + i * 54;
        let cur = null;
        r.at.forEach(([yr, txt]) => { if (year >= yr) cur = [yr, txt]; });
        g.font = 'bold ' + FONT; g.fillStyle = C.text;
        g.fillText(r.n, 34, y + 12);
        g.font = MONO;
        if (!cur) {
          g.fillStyle = '#2a3444';
          g.fillText('— nothing worth the name yet —', MIDX, y + 12);
        } else {
          const latest = r.at[r.at.length - 1][0];
          const isLatest = cur[0] === latest;
          if (isLatest) solved++;
          g.fillStyle = isLatest ? C.green : C.warn;
          wrapLines(g, cur[1], MIDW).slice(0, 2).forEach((ln, j) => g.fillText(ln, MIDX, y + 12 + j * 15));
          g.fillStyle = C.line; g.textAlign = 'right';
          g.fillText('since ' + cur[0], SINCEX, y + 12);
          g.textAlign = 'left';
        }
        g.strokeStyle = C.line; g.lineWidth = 1;
        g.beginPath(); g.moveTo(34, y + 34); g.lineTo(686, y + 34); g.stroke();
      });
      g.font = FONT; g.fillStyle = C.muted;
      wrapText(g, year < 2012
        ? 'Before 2012, every one of these was a separate research field with its own techniques, its own conferences and its own decade-long roadmap.'
        : year < 2020
          ? 'Deep learning is eating the specialised techniques one field at a time, and each conquest still needs its own architecture.'
          : 'One architecture now does all five, and the remaining differences are mostly data and post-training. That convergence is the real story of the last decade.',
        34, 332, 650, 16);
      ro.set({ year: Math.round(year), 'at the frontier': solved + ' / ' + ROWS.length });
    });

    return ctx.figure(cv,
      'Drag the year and watch five separate research fields collapse into one. In 1998 each row had its own techniques, its own conferences and its own decade-long roadmap; by 2022 a single architecture does all of them, and the differences between them are mostly data and post-training. Anyone who tells you they saw that convergence coming in 2010 is misremembering.',
      [ySl, ...presets], ro);
  }

  /* ------------------------------------------------------------------ */
  /* Interactive: what a fixed capability costs over time                */
  /* ------------------------------------------------------------------ */
  function buildCapabilityPrice(ctx) {
    const [cv, g] = ctx.canvas(720, 364);
    const C = ctx.colors;
    const FONT = '13px Inter, system-ui, sans-serif';
    const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
    let year = 2019;
    /* Cost to reach GPT-2 (1.5B) quality, roughly, on the hardware and software of each year.
       Order-of-magnitude only — the point is the slope, not any single figure. */
    const POINTS = [[2019, 50000], [2020, 20000], [2021, 6000], [2022, 2000], [2023, 700], [2024, 200], [2025, 60], [2026, 30]];
    const costAt = (y) => {
      for (let i = 0; i < POINTS.length - 1; i++) {
        const [y0, c0] = POINTS[i], [y1, c1] = POINTS[i + 1];
        if (y >= y0 && y <= y1) {
          const t = (y - y0) / (y1 - y0);
          return Math.exp(Math.log(c0) + t * (Math.log(c1) - Math.log(c0)));
        }
      }
      return POINTS[POINTS.length - 1][1];
    };
    const ySl = ctx.slider({ label: 'year', min: 2019, max: 2026, step: 0.25, value: 2019, fmt: (v) => v.toFixed(2).replace(/\.00$/, ''), onChange: (v) => { year = v; } });
    const ro = ctx.readout();

    ctx.loop(() => {
      g.clearRect(0, 0, cv.W, cv.H);
      const P = { x: 70, y: 50, w: 400, h: 190 };
      const R = 7;                       /* the marker's radius, kept inside the frame */
      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      g.fillText('what it costs to train a GPT-2-class model', P.x, 30);
      g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
      /* the year range is inset by one marker radius so the dot at either end
         sits whole inside the plot rather than half-eaten by the clip */
      const px = (y) => P.x + R + (y - 2019) / 7 * (P.w - 2 * R);
      const py = (c) => P.y + P.h - (Math.log10(c) - 1) / 4 * P.h;
      g.font = MONO; g.fillStyle = C.muted; g.textAlign = 'right';
      [10, 100, 1000, 10000, 100000].forEach(c => {
        g.beginPath(); g.moveTo(P.x, py(c)); g.lineTo(P.x + P.w, py(c)); g.stroke();
        g.fillText('$' + (c >= 1000 ? (c / 1000) + 'k' : c), P.x - 8, py(c) + 4);
      });
      g.textAlign = 'center';
      [2019, 2021, 2023, 2025].forEach(y => g.fillText(String(y), px(y), P.y + P.h + 18));
      g.textAlign = 'left';
      /* everything that follows from the data is clipped to the plot box */
      g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
      g.strokeStyle = C.green; g.lineWidth = 2.5; g.beginPath();
      for (let i = 0; i <= 70; i++) { const y = 2019 + i / 70 * 7; i ? g.lineTo(px(y), py(costAt(y))) : g.moveTo(px(y), py(costAt(y))); }
      g.stroke();
      const c0 = costAt(year);
      g.fillStyle = C.accent;
      g.beginPath(); g.arc(px(year), py(c0), 6, 0, 7); g.fill();
      g.restore();
      /* the note lives in the empty bottom-left corner, clear of the curve and
         of the marker's starting position */
      g.font = MONO; g.fillStyle = C.line;
      g.fillText('log scale — each line is 10×', P.x + 8, P.y + P.h - 10);

      const TX = 510;
      g.font = FONT; g.fillStyle = C.muted; g.fillText('in ' + year.toFixed(2).replace(/\.00$/, '') + ' it costs', TX, 70);
      g.font = 'bold 30px Inter, system-ui, sans-serif'; g.fillStyle = C.green;
      g.fillText('$' + Math.round(c0).toLocaleString(), TX, 106);
      g.font = MONO; g.fillStyle = C.muted;
      g.fillText('down ' + Math.round(50000 / c0) + '× since 2019', TX, 130);
      g.font = FONT; g.fillStyle = C.muted;
      wrapText(g, 'Same capability. Better hardware, better kernels, better recipes, and a far better understanding of how much data to use.', TX, 156, 180, 16);
      /* two columns under the chart, with a gutter wide enough that the bold
         statement on the left cannot run into the paragraph on the right */
      g.font = 'bold ' + FONT; g.fillStyle = C.text;
      wrapLines(g, 'in 2019 this was the most capable language model on Earth.', 200)
        .forEach((ln, i) => g.fillText(ln, 34, 292 + i * 18));
      g.font = FONT; g.fillStyle = C.muted;
      wrapText(g, 'Order-of-magnitude figures only. The slope is the point: a capability that is a research milestone one year is a hobby project a few years later, and that is the clearest reason to be careful about calling anything permanently out of reach.', 262, 292, 420, 16);
      ro.set({ year: year.toFixed(2).replace(/\.00$/, ''), cost: '$' + Math.round(c0).toLocaleString(), 'cheaper than 2019': Math.round(50000 / c0) + '×' });
    });

    return ctx.figure(cv,
      'GPT-2 was the most capable language model in the world in 2019 and its staged release was a public argument about AI risk. Today the same capability is reproducible for the price of a dinner — Karpathy\'s llm.c reproduces GPT-2 on eight A100s for about twenty dollars of rented time. These are order-of-magnitude figures, and the slope is what matters: capability gets dramatically cheaper even when nothing new is invented.',
      [ySl], ro);
  }

  ZTA.registerChapter({
    id: '14-timeline',
    num: 14,
    part: 'V',
    title: '80 years in one scroll: the history of AI',
    tagline: 'From a 1943 paper about neurons to models that write code for hours: sixty moments, two winters, and the five things that had to arrive at once.',
    render(root, ctx) {
      const h = ctx.h;
      const legend = h('div', { style: { display: 'flex', gap: '14px', flexWrap: 'wrap', margin: '0 0 1em', fontSize: '.85rem' } },
        ...TAGS.map(t => h('span', { style: { display: 'inline-flex', alignItems: 'center', gap: '6px', color: t.color } }, h('span', { style: { width: '10px', height: '10px', borderRadius: '50%', background: t.color, display: 'inline-block' } }), t.label)));

      const list = h('div', { class: 'timeline' }, ...ENTRIES.map(e => h('div', { class: 'tl-item' },
        h('div', { class: 'tl-year' }, e.year + '  ·  ' + e.tag),
        h('div', { class: 'tl-title' }, e.title),
        h('div', { class: 'tl-desc', html: e.desc + (e.scale ? ' <span class="stat">' + fmtBig(e.scale) + ' parameters.</span>' : '') }))));

      root.append(
        ctx.callout('tryit', '🖐 Do this first — scroll eighty years in ten seconds',
          '<b>1.</b> Start zoomed out and look at the green curve. It is <b>flat for fifty years</b>, then climbs ten orders of magnitude in fourteen.<br>' +
          '<b>2.</b> Find the two gaps where almost nothing happens. Those are the AI winters, and the field ran out of money and credibility in both.<br>' +
          '<b>3.</b> Now zoom into 2012-2026 and watch the same curve turn near-vertical.<br>' +
          '<b>4.</b> Click any entry to read what actually happened.'),
        buildTimeline(ctx),
        ctx.p('Almost everything you use was built in the last sliver on the right. Almost every idea it rests on is decades older.'),

        ctx.p('In July 1958 the New York Times reported that the US Navy had unveiled an electronic brain, the perceptron, that was expected to "walk, talk, see, write, reproduce itself and be conscious of its existence". '),
        ctx.p('It could, in fact, learn to tell a card marked on the left from a card marked on the right. Sixty-four years later, in November 2022, a chat window went online that a hundred million people were using within two months. Between those two moments lie two funding collapses, several rebrandings, a handful of stubborn people who kept working when it was unfashionable, and a chip designed to make video games look better.'),
        ctx.p('History is the part of a technical subject people skip, and for AI that is a mistake, because the story explains the shape of the present. Why does every model use a transformer? Because a 2017 paper made it possible to use GPUs fully. Why do labs count tokens instead of parameters? Because of a 2022 DeepMind paper. Why is everyone nervous about hype? Because the field has been through it twice before and both times the money vanished for a decade.'),
        ctx.p('This chapter is one scroll through eighty years. The question it answers: <b>what actually had to happen, and in what order, for a machine to learn to talk?</b> The timeline below is interactive; every dot is a story. Below it, the same entries as a plain list you can read straight through.'),

        ctx.section('The timeline',
          ctx.p('Each entry is tagged by what kind of thing it was. <em>Theory</em> is an idea on paper; <em>architecture</em> is a new shape of network; <em>hardware</em> and <em>data</em> are the raw materials; <em>product</em> is when it reached people; <em>milestone</em> is a result that changed what everyone believed possible; <em>winter</em> is when the money left.'),
          legend,
        ),

        ctx.section('The exponential, made visceral',
          ctx.p('The green curve deserves a second look. Rosenblatt\'s perceptron had a few hundred adjustable weights. LeNet-5 had sixty thousand. AlexNet, sixty million. GPT-3, 175 billion. The largest models of 2026 are estimated at around a trillion. That is ten orders of magnitude: not "a lot bigger" but bigger in the way a galaxy is bigger than a grain of sand. And the curve of training <em>compute</em>, the number of arithmetic operations spent on a single training run, is steeper still, because compute grows with both model size and data.'),
          ctx.p('In 2018 OpenAI measured that the compute used in the largest training runs had doubled every 3.4 months between AlexNet (2012) and AlphaGo Zero (2017), a 300,000-fold increase in six years, '),
          ctx.p('That was far faster than Moore\'s law, because it was driven by spending and parallelism rather than by transistor density. Epoch AI\'s more recent estimates put frontier training compute growth at roughly four to five times per year since 2010, a doubling every six months or so. Both are approximate; neither has clearly stopped. The explorer below lets you feel what those numbers mean.'),
          ctx.callout('tryit', 'Try it', 'Leave the start at 2012 and set the doubling time to 3.4 months: read off the multiplier at "today". Now set it to 24 months (Moore\'s law). The gap between the two lines at 2026 is the difference between a computer that is a few hundred times faster and one that is a hundred billion times more compute-hungry. Then move the start to 2020 and ask: how many doublings since GPT-3?'),
          buildDoubling(ctx),
        ),

        ctx.section('Winters and springs',
          ctx.callout('tryit', '🖐 Try this — guess when the experts were right',
            '<b>1.</b> Read each prediction, drag the slider to when you think it actually happened, then press <b>Reveal</b>.<br>' +
            '<b>2.</b> Two were decades late, one was eight years early, and two have still not happened.<br>' +
            '<b>3.</b> Every one was made by a leading expert with full knowledge of the state of the art. <b>That is the point.</b>'),
          buildPredictionGame(ctx),

          ctx.p('Twice, the field has run out of money and reputation at the same time. '),
          ctx.p('The first winter (roughly 1974 to 1980) followed a decade of promises: machine translation that would be solved in a few years, programs that would be world chess champions by 1968, "general problem solvers". The programs worked on toy problems and fell apart on real ones, because the number of possibilities to search explodes with problem size and nobody had the data or compute to learn instead of search. Governments in the UK and US read damning reports and cut funding. Researchers renamed their work.'),
          ctx.p('The second winter (roughly 1987 to 1993) followed the expert-systems boom. Rule-based systems had genuinely saved companies money, but every rule had to be typed by a human, the systems could not learn, and they broke at the edges of what their authors had anticipated. When cheaper hardware killed the specialised Lisp machines the industry ran on, and Japan\'s Fifth Generation project ended without its goals, "AI" became a word that made investors leave the room. That is when "machine learning" became the polite term.'),
          ctx.p('What is different this time? The honest answer has three parts. First, the current systems make money: coding assistants, search, customer service, image generation are products with revenue, not demos. '),
          ctx.p('Second, the core method, learning from data, does not have the brittleness that killed expert systems; it degrades gracefully instead of falling off a cliff. '),
          ctx.p('Third, and least comfortably: the scale of investment is now so large (hundreds of billions of dollars a year in chips and data centres) that a disappointment would be a different kind of event. Whether a third winter is possible is a serious question; chapter 15 takes it seriously. What history says is that winters come from the gap between promise and delivery, so the best insurance is to promise accurately.'),
          ctx.callout('history', 'The people who kept going', 'Geoffrey Hinton moved to Canada in 1987 partly because he did not want US military funding, and spent the second winter at the University of Toronto on grants from a small Canadian institute. Yann LeCun kept building convolutional networks at Bell Labs while the field ignored them. Yoshua Bengio worked on neural language models in Montreal when almost nobody else did. Jürgen Schmidhuber\'s group in Switzerland kept LSTMs alive for a decade of indifference. In 2018 Hinton, LeCun and Bengio shared the Turing Award; in 2024 Hinton shared the Nobel Prize in Physics. The lesson is not that persistence always wins. It is that the winners of the spring were the people who had been right during the winter.'),
        ),

        ctx.section('What was actually possible, year by year',
          ctx.p('The timeline says what was invented. This says what you could actually do with it.'),
          ctx.callout('tryit', '🖐 Try this',
            '<b>1.</b> Press <b>1998</b>. Five separate research fields, five separate sets of techniques, five separate decade-long roadmaps.<br>' +
            '<b>2.</b> Step through <b>2014</b>, <b>2020</b>, <b>2025</b> and watch them collapse into one architecture.<br>' +
            '<b>3.</b> Note how recent the last column is. Most of it is inside the last four years.'),
          buildStateOfArt(ctx),
          ctx.p('And capability does not only arrive — it gets cheap, fast, often without anything new being invented.'),
          ctx.callout('tryit', '🖐 Try this',
            'Drag the year and watch what it costs to train a GPT-2-class model. In 2019 this was the most capable language model on Earth and its release was a public argument about AI risk. Today it is a weekend project.'),
          buildCapabilityPrice(ctx),
        ),

        ctx.section('The five ingredients that had to arrive together',
          ctx.p('Backpropagation was published in 1986. Convolutional networks worked in 1989. Why did the revolution wait until 2012? Because a working method was only one of five things needed, and the other four arrived on their own schedules.'),
          ctx.cards([
            { title: '1 · Algorithms', body: 'Backprop (1986), convolutions (1989), LSTMs (1997), ReLU and dropout (2010–12), attention (2014), the transformer (2017), RLHF (2022). Each removed one obstacle to training bigger models on messier data. None was sufficient alone.' },
            { title: '2 · Data', body: 'The web. ImageNet\'s 14 million labelled photos (2009) made AlexNet possible; Common Crawl\'s petabytes of text made GPT-3 possible. Before the internet there was simply not enough recorded human behaviour to learn from.' },
            { title: '3 · Compute', body: 'GPUs built for games (1999), made programmable (2007), then designed for tensors (2017 onward). A 2012 gaming card did in a week what a 1990 workstation would have needed decades for. Compute per dollar for this kind of arithmetic fell about a thousandfold between 2000 and 2020.' },
            { title: '4 · Money', body: 'Google buying DeepMind (2014), Microsoft\'s billions into OpenAI (2019–2023), and eventually the largest capital expenditure programme in the history of technology. Frontier runs cost hundreds of millions; nobody funds that without a proven business.' },
            { title: '5 · Open publishing culture', body: 'arXiv preprints, open-source frameworks (Theano, Torch, TensorFlow, PyTorch), public benchmarks, released weights. Ideas crossed labs in days. The transformer was reproduced worldwide within months of publication. This ingredient is the one most at risk in 2026, as frontier labs publish less.' },
          ]),
          ctx.p('Take any one away and the story stalls. Algorithms without compute: the 1990s. Compute and data without a good algorithm for using them: the search-based approaches of the 1970s. Everything but money: a great paper nobody scales. Everything but openness: five companies reinventing each other\'s work in secret, slowly. The 2012 to 2022 decade is what happens when all five click, and the reason it felt sudden is that four of the five had been quietly maturing for years.'),
        ),

        ctx.callout('example', 'Where each era lives in your pocket', 'The keyboard autocorrect on your phone descends from the n-gram and LSTM language models of the 1990s and 2010s. Face unlock is a convolutional network, a direct descendant of LeNet. Voice assistants of the 2010s were LSTMs; the ones of the 2020s are transformers. Your photo app\'s search box is CLIP or a cousin. And the chat assistant is InstructGPT\'s recipe on a transformer trained with GPT-3\'s scaling, tuned with Constitutional AI or RLHF, running on Hopper or Blackwell chips. Every dot on the timeline is still running somewhere.'),

        ctx.section('Why it matters for modern AI',
          ctx.p('Three things to take from eighty years. First, progress in AI has never been smooth: it comes in bursts when ingredients align, followed by plateaus that look permanent from inside. Anyone claiming to know the schedule for the next burst is guessing. '),
          ctx.p('Second, the ideas that won were mostly old; what changed was the ability to run them at scale. That argues for taking "impractical" ideas seriously and for taking hardware seriously. Third, the field has repeatedly mistaken a capability for a solved problem (chess, then Go, then conversation) and each time the goalposts moved for good reasons: the capability turned out to be narrower than it looked. Chapter 15 asks what is left, and whether the current burst reaches all the way.'),
        ),

        ctx.quiz([
          { q: 'Put these in chronological order: the Transformer paper, backpropagation popularised, the Perceptron, AlexNet.', options: ['Perceptron → backprop → AlexNet → Transformer', 'Backprop → Perceptron → Transformer → AlexNet', 'Perceptron → AlexNet → backprop → Transformer', 'AlexNet → Perceptron → backprop → Transformer'], answer: 0, explain: 'Perceptron 1958, backprop popularised 1986, AlexNet 2012, Transformer 2017. Twenty-six years passed between backprop and the deep-learning breakthrough, waiting for data and compute.' },
          { q: 'What combination made AlexNet\'s 2012 result possible?', options: ['A new learning algorithm nobody had seen', 'A large labelled dataset (ImageNet), gaming GPUs made programmable by CUDA, and a deep convolutional network', 'Google\'s TPUs', 'The transformer architecture'], answer: 1, explain: 'The algorithm (convolutions + backprop) was from 1989. ImageNet (2009) and CUDA-programmable GPUs (2007) were the new ingredients; two GTX 580 cards trained it in about a week.' },
          { q: 'What did the two AI winters have in common?', options: ['A shortage of researchers', 'Promises that outran what the systems could deliver on real problems, followed by funding cuts', 'Government bans on AI', 'A lack of interest from the military'], answer: 1, explain: 'Both winters followed periods of over-promising: search-based programs that failed to scale (1970s) and expert systems that were brittle and costly to maintain (late 1980s). Money and reputation left together.' },
          { q: 'Which is the correct pairing?', options: ['2017 — GPT-3; 2020 — Transformer', '2017 — Transformer; 2020 — GPT-3 and scaling laws', '2012 — Transformer; 2017 — AlexNet', '2022 — GPT-3; 2017 — ChatGPT'], answer: 1, explain: '"Attention Is All You Need" is June 2017; GPT-3 and Kaplan\'s scaling-laws paper are both 2020; ChatGPT launched 30 November 2022.' },
          { q: 'Roughly how fast has compute for the largest training runs grown since 2012?', options: ['Doubling every two years, like Moore\'s law', 'Doubling every few months: about 3.4 months in 2012–2018 per OpenAI, ~6 months since per Epoch AI', 'It has stayed flat', 'Doubling every ten years'], answer: 1, explain: 'Training compute grew far faster than chip density because it is driven by spending and parallelism, not just transistors. Both figures are approximate estimates from published analyses.' },
        ]),

        ctx.section('Go deeper',
          ctx.ul([
            '<a href="https://openai.com/research/ai-and-compute" target="_blank">OpenAI — AI and Compute (2018)</a>: the 3.4-month doubling analysis, with the chart that started the conversation.',
            '<a href="https://epoch.ai/trends" target="_blank">Epoch AI — Trends</a>: continually updated data on training compute, model sizes, hardware and data across the whole history of ML.',
            '<a href="https://academic.oup.com/mind/article/LIX/236/433/986238" target="_blank">Turing (1950) — Computing Machinery and Intelligence</a>: still the best essay on the subject; read the objections section.',
            '<a href="https://www.nature.com/articles/nature14539" target="_blank">LeCun, Bengio, Hinton (2015) — Deep Learning (Nature review)</a>: the three pioneers summarise the field at the moment it took off.',
            '<a href="https://karpathy.medium.com/software-2-0-a64152b37c35" target="_blank">Karpathy — Software 2.0 (2017)</a>: an argument that learned programs are a new kind of software, written just before the transformer era.',
            '<a href="https://en.wikipedia.org/wiki/History_of_artificial_intelligence" target="_blank">Wikipedia — History of artificial intelligence</a>: thorough, well-sourced, and good on the winters.',
          ]),
        ),

        ctx.section('Every entry, as a list',
          ctx.p('The same ' + ENTRIES.length + ' entries as the canvas, in order, for reading straight through or for screen readers.'),
          list,
        ),
      );
    },
  });
})();
