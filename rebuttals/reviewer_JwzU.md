<!-- REVIEWER JwzU — Overall 2 ("resubmit next cycle"), Confidence 3, Soundness 3. -->

**Response to Reviewer JwzU**

Thank you for the careful review. Your concerns — task coverage, template/lexical confounds, and the partial cross-model replication — are exactly the right ones to press on, and we spent the response period addressing them with new experiments. We believe the substantial revisions your assessment calls for have been completed within this cycle, and we address each concern below.

---

**W1: Over-refusal appears in only two tasks; the dataset is small (especially for causal patching).**
**Comment: "Can the authors evaluate more tasks where over-refusal actually occurs?"**

> **Response:** Yes — we added **four new task frames** over a shared pool of 205 contents drawn from the original benign tasks (+820 samples; 1,090 total), using the paper's exact pipeline:

> | New task frame | Over-refusals | Rate |
> |---|---|---|
> | Conversational ("reply helpfully to this user message: …") | **75 / 205** | **37%** |
> | POS tagging | 3 / 205 | 1.5% |
> | Character shuffle | 3 / 205 | 1.5% |
> | Word shuffle | 1 / 205 | 0.5% |

> The conversational frame is a task "where over-refusal actually occurs" — at a larger scale than any task in the paper (previous maximum: translation, n=28). The three mechanical frames elicit almost none, despite wrapping the *same contents*. This split itself supports the task-conditioned account: over-refusal fires where the model must semantically engage with content, and vanishes where text is merely transformed. With this third population, per-task over-refusal directions align at **0.428 ± 0.058** (three tasks), vs. **0.536 ± 0.086** for harmful-refusal directions computed identically on the same data — over-refusal remains measurably the more task-specific of the two. We will extend §4.2 and Table 1 accordingly.

> On the small-n concern: bootstrap resampling (2,000×) gives cos(over-refusal dir, harmful-refusal dir) = **+0.448 (95% CI [+0.349, +0.530])** — the CI excludes 1.0 by a wide margin, so the two directions are statistically distinct rather than a sampling artefact.

> Your concern also led us to re-examine the subspace-dimensionality claim (§4.2, Fig. 6) under a sample-size-matched control, since the original 11-vs-8 comparison used unequal sample sizes (n=48 vs 25). Across **four independent conditions** (original population, enlarged, task-balanced, largest-task-excluded), the probability that over-refusal needs more components than harmful-refusal is at most 0.9%. **The claim does not survive this control, and we will revise or remove it in the revision.** We thank you for prompting the scrutiny that surfaced this.

> On causal patching, we agree: we will reframe §4.6 as negative evidence only (no single bottleneck head) and extend to 20 contrastive pairs per head.

---

**W2: Potential confounding from task templates and dataset differences; early-layer separability may be partly lexical; more controlled prompt pairs would strengthen the claims.**

> **Response:** The new-task experiment was designed to enable exactly these controls:

> **(1) Templates.** Each new frame uses three paraphrased instruction templates; the template-variant silhouette is **+0.03** (near zero) — phrasing contributes essentially nothing to the geometry.

> **(2) Content.** Identical contents appear in every frame, so same-content/different-frame pairs are lexically near-identical controlled prompts. A task probe evaluated **only on held-out contents** (GroupKFold by content id, 8-way, chance 12.5%) reaches **98.2% ± 1.1** balanced accuracy — task identity is genuinely encoded, independently of both content and phrasing.

> **(3) The early-layer lexical point — partially conceded.** We attempted a direct test on the XSTest minimal pairs in our evaluation set, but only 2 refused safe-twins exist — too few to support a probe, and we do not claim otherwise. We therefore cannot rule out a lexical contribution to the *early-layer* refusal-type separability; we will scope that observation in §4.3 and add a properly-powered XSTest minimal-pair analysis in the revision. We note the paper's geometric claims rest on the *mid-layer* task structure, which controls (1)–(2) address on controlled pairs.

---

**W3: The cross-model replication is partial — Qwen does not validate the central directional comparison.**

> **Response:** The reason is a property of the model, not the analysis: on the same 270 prompts, Qwen1.5-7B produces exactly **one** harmful refusal, leaving its harmful-refusal direction undefined. Everything computable on Qwen does replicate (task constellations, within-cluster placement of over-refusal, probe ordering). We agree the directional claims need a genuine second model: the extended pipeline is model-parameterized, and we are re-running it on a model with a non-trivial harmful-refusal rate; we commit to the full cross-model directional comparison in the revision.

---

We believe these additions — more over-refusal tasks (your explicit request), template/content confound controls, statistical validation, and one honest retraction — are the substantial revisions your assessment called for, completed within this cycle. Together with the dataset and code release we are preparing, we hope this addresses your concerns, and we would be grateful if you would consider updating your assessment.
