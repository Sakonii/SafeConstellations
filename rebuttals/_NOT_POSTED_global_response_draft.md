<!-- NOT FOR POSTING. Decision (2026-07-11): no general response will be posted. Reviewers
     don't see each other's threads and a shared response reads as untailored, so every
     point below has been folded directly into the relevant reviewer_*.md file(s). Kept
     here only as an internal working reference for how the content was originally
     organized; see reviewer_JwzU.md, reviewer_xCj3.md, reviewer_Zici.md for the actual
     text to post. -->

**General Response to All Reviewers**

We thank all reviewers for the careful and constructive reviews. We are encouraged that all three reviews find the harmful-refusal vs. over-refusal distinction meaningful and the geometric account insightful. We ran new experiments during the response period for each of the main concerns. We summarize them here and give details in the individual responses.

---

**1. More tasks where over-refusal occurs** (JwzU W1; Zici W1)

We added seven new task frames over a shared pool of 205 contents drawn from all four original benign tasks, reusing the paper's exact pipeline (same hooks, judging protocol, and masks), then confirmed the result at much larger scale using the dataset's own training split (roughly 505 additional contents per task, not used in any of the paper's experiments):

| New task frame | Pilot (205 contents) | Full scale (~645 contents) |
|---|---|---|
| Continuation ("write a continuation of this passage") | **78 (38%)** | **306 (48%)** |
| Conversational ("reply helpfully to this user message: ...") | **75 (37%)** | **318 (49%)** |
| Message drafting ("compose an email or note based on this") | 47 (23%) | 184 (29%) |
| Keyword identification | 39 (19%) | 155 (24%) |
| POS tagging | 3 (1.5%) | not expanded |
| Character shuffle | 3 (1.5%) | not expanded |
| Word shuffle | 1 (0.5%) | not expanded |

The split is sharp: frames that require the model to semantically engage with the content (continuing it, replying to it, drafting from it, identifying its key ideas) elicit substantial over-refusal that holds or rises at scale, while frames that merely transform the same contents structurally stay near zero throughout. The conversational frame was added per Reviewer Zici's suggestion; continuation exceeds even it, and both exceed every task in the paper (previous maximum: translation, n=28). The contrast is not about output length: keyword identification is a short labeling task like sentiment analysis and both trigger substantial over-refusal, while POS tagging (also labeling, but purely syntactic) does not. This gives the directional analysis **six** genuine over-refusal populations instead of the paper's two, computed on the full expanded dataset: per-task over-refusal directions align with each other at **0.358 ± 0.235**, versus **0.636 ± 0.140** for the same tasks' harmful-refusal directions. Over-refusal remains clearly the more task-specific of the two, and the contrast is wider than in the two-task version of this analysis. We also checked whether the three generative tasks (continuation, drafting, conversational) over-count over-refusal on content where compliance could itself be harmful; even excluding all such content, over-refusal remains substantial for all three (details in the response to JwzU, W2).

**2. Template / content / lexical confounds** (JwzU W2; xCj3 suggestion)

The four frames of the first extension round each ran with three paraphrased instruction templates. Phrasing contributes essentially nothing to the geometry (template-variant silhouette **+0.03**, near zero). Because identical contents appear in every frame, same-content pairs across frames are lexically near-identical controlled prompts, and a task probe evaluated only on held-out contents (GroupKFold by content, 8-way, chance 12.5%) still reaches **98.2%** balanced accuracy. Task identity is therefore encoded independently of both content and phrasing. We note one limitation: our evaluation set contains too few refused XSTest safe-twins (n=2) to probe the early-layer lexical question, so we cannot rule out a lexical contribution to early-layer refusal-type separability. We will scope that claim in the revision and add a properly-powered XSTest minimal-pair analysis.

**3. Statistical support, including one claim we retract** (JwzU W1)

Bootstrap resampling (2,000 resamples) places cos(over-refusal dir, harmful-refusal dir) at **+0.448 (95% CI [+0.349, +0.530])**. The CI excludes 1.0 by a wide margin, so the two directions are statistically distinct and not a small-sample artifact. The inter-task silhouette is similarly stable (0.357, 95% CI [0.325, 0.409]), and a layer-neighborhood sweep (L9 to L15) shows broad plateaus. However, the subspace-dimensionality claim (§4.2, Fig. 6) does not survive a sample-size-matched control. The original 11-vs-8 comparison used unequal sample sizes (n=48 vs 25), and across four independent matched-n conditions (original population, enlarged, task-balanced, largest-task-excluded) the probability that over-refusal needs more components is at most 0.9%. **We will revise or remove this claim in the revision.** We believe this correction is a direct benefit of the review's scrutiny.

**4. Independent validation of the refusal labels** (xCj3 W1)

A second judge from a different model family (Claude) agrees with GPT-4o on **95.2%** of 3-class labels (κ = 0.899) and **97.6%** of the collapsed refusal-vs-answer labels (κ = 0.947), which is the only distinction the paper's group definitions use. 60 of the 270 samples contain genuinely harmful content that triggered the second judge's own safety classifier; these are excluded rather than substituted, and we note this in passing as a small illustration of the paper's subject matter.

**5. Artifacts** (Zici, JwzU: Datasets/Software ratings)

We will release the evaluation dataset and the complete notebook pipeline (dataset construction, embedding extraction, all analyses, and the new experiments above) with the camera-ready version.

**6. Cross-model directional replication** (JwzU W3)

Qwen1.5-7B (used in the paper's appendix) produces exactly one harmful refusal on our 270 prompts, an artifact of that model's low refusal rate rather than a limitation of the analysis. We re-ran the full pipeline on Qwen2.5-7B-Instruct, which has stronger safety tuning, on the original prompts plus all seven new task frames (1,705 total): 270 over-refusals and 73 harmful-refusals, both well above the minimum needed for a direction estimate, so the central directional comparison is now computable for the first time on a second model. cos(over-refusal dir, harmful-refusal dir) = **+0.784 (95% CI [+0.724, +0.823])**, excluding perfect alignment: the two directions remain distinct, though more aligned than on LLaMA (+0.448, point 3). The task-conditioned split itself replicates cleanly (mechanical tasks: 0-4 over-refusals per 205 prompts; generative tasks: 41-77, the same pattern as on LLaMA). Full details, including a note on the model-dependence of the already-retracted dimensionality claim, are in the response to JwzU.

---

We also commit to: reframing causal patching as negative evidence with 20 pairs per head (xCj3 W2, JwzU W1), and scoping the abstract and introduction claims to the models and tasks studied (xCj3 W3). One sub-claim (subspace dimensionality) is retracted on LLaMA specifically. The paper's central claim is unaffected and now rests on six task frames instead of two, replicated on a second model family: harmful-refusal is task-agnostic, over-refusal is task-conditioned, and interventions must respect that asymmetry.
