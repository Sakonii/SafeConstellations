<!-- REVIEWER xCj3 — Overall 3.5, Confidence 4, Soundness 4 (expert; strongest reviewer). -->

**Response to Reviewer xCj3**

Thank you for the positive and expert assessment, and for the concrete, actionable suggestions — we implemented all three during the response period.

---

**W1: Refusal labels rely on GPT-4o without independent validation; a human-checked subset or second-judge agreement would strengthen the paper.**

> **Response:** We re-labelled all 270 responses with a **second judge from a different model family** (Claude, identical OR-Bench 3-class prompt, temperature 0), rather than a same-vendor model, so the check is genuinely independent:

> | Agreement with GPT-4o | Value |
> |---|---|
> | 3-class labels | **95.2%** (κ = 0.899) |
> | Collapsed refusal-vs-answer | **97.6%** (κ = 0.947) |

> The collapsed number is the operative one: all group definitions in the paper (over-refusal / harmful-refusal / harmless-answered) use only refusal-vs-answer, so only collapsed disagreements can move a sample between groups — and at 97.6%, potential label noise sits well inside the bootstrap confidence intervals we report for the geometric quantities (General Response, point 3).

> One aside: 60 of the 270 samples (22%) contain genuinely harmful content that triggered the second judge's own safety classifier even for this pure classification task; these are excluded from the agreement figures rather than patched with a same-vendor substitute. We note this as a small, independent illustration of the paper's subject matter. We will add the agreement table to the appendix, and we are preparing a human-annotated validation of a stratified subset for the camera-ready version.

---

**W2: Causal patching should be interpreted cautiously — the results support the absence of a single bottleneck head, not a circuit-level localization claim.**

> **Response:** We agree, and will reframe §4.6 exactly along these lines: the patching results are **negative evidence** — no single head crosses the 50% flip threshold — and we will remove the wording suggesting positive localization ("confirms that refusal is mediated by distributed, multi-layer, task-specific circuits" → "is consistent with a distributed implementation; identifying the actual circuit is left to future work"). We are also increasing the contrastive pairs from 5 to 20 per head so the negative result is properly powered.

---

**W3: Be more cautious about generalization in the abstract and introduction; consider narrowing the title.**

> **Response:** We will scope the claims explicitly: the abstract and introduction will state that the analysis covers instruction-tuned models at the 7–8B scale (LLaMA-3.1-8B, partial replication on Qwen1.5-7B) across eight NLP task frames, and that the geometric account is a property of the models and tasks studied rather than a universal claim. We are also open to narrowing the title (e.g., "…in Aligned LLMs" → "…in Instruction-Tuned LLMs") and would welcome your preference.

---

**On more tightly controlled harmful vs. sensitive-benign prompt pairs:**

> **Response:** Our new tasks are **content-matched by construction** — the same 205 content strings appear in every task frame, so same-content/different-frame pairs are lexically near-identical controlled prompts, and a task probe evaluated only on held-out contents (GroupKFold by content, 8-way) reaches **98.2%** balanced accuracy (General Response, point 2). For refusal-type minimal pairs specifically, our evaluation set contains too few refused XSTest safe-twins (n=2) to support a probe; we will add a properly-powered minimal-pair analysis (the full XSTest set) to the revision.

---

We thank you again for the constructive review. All three concerns translate into direct improvements — independent label validation, honestly-scoped patching claims, and a properly-scoped abstract — and we will incorporate them, along with the new controlled-pair experiments, into the revision.
