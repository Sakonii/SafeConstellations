<!-- GLOBAL RESPONSE — post once as the general author response. -->

**General Response to All Reviewers**

We thank all reviewers for the careful and constructive reviews, and are encouraged that all three find the harmful-refusal vs. over-refusal distinction meaningful and the geometric account insightful. During the response period we ran new experiments for each main concern, summarized here; details are in the individual responses.

---

**1. More tasks where over-refusal occurs** (JwzU W1; Zici W1)

We added **four new task frames** over a shared pool of 205 contents drawn from all four original benign tasks (+820 samples; 1,090 total), reusing the paper's exact pipeline (same hooks, judging protocol, and masks):

| New task frame | Over-refusals | Rate |
|---|---|---|
| Conversational ("reply helpfully to this user message: …") | **75 / 205** | **37%** |
| POS tagging | 3 / 205 | 1.5% |
| Character shuffle | 3 / 205 | 1.5% |
| Word shuffle | 1 / 205 | 0.5% |

The conversational frame (added per Reviewer Zici's suggestion) elicits more over-refusal than any task in the paper (previous maximum: translation, n=28), while mechanical transformations of the *same contents* elicit almost none. Over-refusal concentrates where the model must semantically engage with content, and disappears where text is treated as a formal object to transform — consistent with the task-conditioned account. This also gives the directional analysis a third genuine over-refusal population: per-task over-refusal directions align at **0.428 ± 0.058** across three tasks, vs. **0.536 ± 0.086** for harmful-refusal directions computed identically on the same data — over-refusal remains the more task-specific of the two.

**2. Template / content / lexical confounds** (JwzU W2; xCj3 suggestion)

Each new frame ran with three paraphrased instruction templates: phrasing contributes essentially nothing (template-variant silhouette **+0.03**, near zero). Because identical contents appear in every frame, same-content/different-frame pairs are lexically near-identical controlled prompts — and a task probe evaluated **only on held-out contents** (GroupKFold by content, 8-way, chance 12.5%) reaches **98.2%** balanced accuracy: task identity is encoded independently of content and phrasing. One honest limitation: our evaluation set contains too few refused XSTest safe-twins (n=2) to probe the *early-layer* lexical question; we will scope that claim and add a properly-powered XSTest minimal-pair analysis in the revision.

**3. Statistical support — including one claim we retract** (JwzU W1)

Bootstrap resampling (2,000×) places cos(over-refusal dir, harmful-refusal dir) at **+0.448 (95% CI [+0.349, +0.530])** — the CI excludes 1.0 by a wide margin, so the two directions are statistically distinct, not a small-n artefact. The inter-task silhouette is similarly stable (0.357, 95% CI [0.325, 0.409]), and a layer-neighborhood sweep (L9–L15) shows broad plateaus. However, the subspace-dimensionality claim (§4.2, Fig. 6) does **not** survive a sample-size-matched control: the original 11-vs-8 comparison used unequal sample sizes (n=48 vs 25), and across four independent matched-n conditions (original population, enlarged, task-balanced, largest-task-excluded) the probability that over-refusal needs more components is at most 0.9%. **We will revise or remove this claim in the revision.** We see this correction as a direct benefit of the review's scrutiny.

**4. Independent validation of the refusal labels** (xCj3 W1)

A second judge from a different model family (Claude) agrees with GPT-4o on **95.2%** of 3-class labels (κ = 0.899) and **97.6%** of the collapsed refusal-vs-answer labels (κ = 0.947) — the only distinction the paper's group definitions use. (60/270 samples with genuinely harmful content triggered the second judge's own safety classifier and are excluded rather than substituted — itself a small illustration of the paper's subject matter.)

**5. Artifacts** (Zici, JwzU: Datasets/Software ratings)

We will release the evaluation dataset and the complete notebook pipeline (dataset construction, embedding extraction, all analyses, and the new experiments above) with the camera-ready version.

---

We also commit to: reframing causal patching as negative evidence with 20 pairs per head (xCj3 W2, JwzU W1), scoping the abstract/introduction claims to the models and tasks studied (xCj3 W3), and completing the cross-model directional comparison on a second model with a non-trivial harmful-refusal rate (JwzU W3). One sub-claim (subspace dimensionality) is retracted; the paper's central claim is unaffected and now rests on three task frames instead of two: harmful-refusal is task-agnostic, over-refusal is task-conditioned, and interventions must respect that asymmetry.
