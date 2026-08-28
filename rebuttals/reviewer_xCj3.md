<!-- REVIEWER xCj3: Overall 3.5, Confidence 4, Soundness 4 (expert; strongest reviewer). -->

**Response to Reviewer xCj3**
Thank you reviewer for the positive assessment and constructive suggestions. We acknowledge each of them as follows:

---

**W1: Refusal labels rely on GPT-4o without independent validation; a human-checked subset or second-judge agreement would strengthen the paper.**

> **Response:** We re-evaluated the responses with a second judge from a different model family (Claude-sonnet-5 using the same OR-Bench prompt), and report the Cohen's Kappa (κ) agreement:

> | Agreement with GPT-4o | Value |
> |---|---|
> | 3-class labels | **95.2%** (κ = 0.899) |
> | 2-class (refusal-vs-answer only) | **97.6%** (κ = 0.947) |

> There seems strong agreement between the models. We manually reviewed a stratified sample of the judged responses and validated that the automated labels are accurate. We will include this discussion in the Appendix section by the camera-ready version.

---

**W2: Causal patching should be interpreted cautiously; the results support the absence of a single bottleneck head, not a circuit-level localization claim.**

> **Response:** We appreciate the reviewer’s careful observation, and will reframe §4.6 accordingly. The patching results are negative evidence (no single head crosses the 50% threshold), and we will correct the wording that suggested positive localization (specifically, "confirms that refusal is mediated by distributed, multi-layer, task-specific circuits" becomes something that suggest, "is consistent with a distributed implementation; identifying the actual circuit is left for future work").



---

**W3: Be more cautious about generalization in the abstract and introduction; consider narrowing the title.**

> **Response:** As suggested by the reviewer, we will scope the claims explicitly. The abstract and introduction will state that the analysis covers models at the 7-8B scale (LLaMA-3.1-8B, with cross-model directional replication on Qwen2.5-7B-Instruct and an additional replication on Qwen1.5-7B), and that the geometric findings are properties of the models and tasks studied rather than a universal claim.

---

We thank you again for the constructive review. The three concerns (independent label validation, cautiously scoped patching claims, and a properly scoped abstract) suggest constructive improvements to our manuscript, which we will incorporate in the camera-ready version.
