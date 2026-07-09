<!-- REVIEWER Zici — Overall 3.5, Confidence 2, Soundness 4. -->

**Response to Reviewer Zici**

Thank you for the positive assessment and the thoughtful reading from outside the immediate subfield. Both weaknesses you raise are fair, and we addressed each with new experiments during the response period.

---

**W1: The five task domains are not very compelling for this question — over-refusal matters most for conversational systems.**

> **Response:** We agree, and we took the suggestion concretely: during the response period we added an open-ended **conversational frame** ("reply helpfully to this user message: …") plus three mechanical frames (POS tagging, word/character shuffling), all over a shared pool of 205 contents. The result is exactly what your suggestion anticipated:

> | Task frame | Over-refusal rate |
> |---|---|
> | Conversational | **37%** (75/205 — more than any task in the paper) |
> | Translation / sentiment analysis | 30% / 24% |
> | POS tagging / char shuffle / word shuffle | 1.5% / 1.5% / 0.5% |

> The closer a task frame comes to "just responding to the user's message," the more over-refusal fires; frames that treat text as a formal object to transform (shuffle, tag) show almost none, despite wrapping the same contents. We will make the conversational frame a full condition in the revision and reframe the task-selection discussion in §3.1 around this dimension — task frames resembling unconstrained conversation are precisely where the phenomenon is most severe.

---

**W2: The layer selections are well motivated but the differences from neighboring layers look minor — would results differ in a small neighborhood around the maximal points?**

> **Response:** No — and we verified this explicitly. All quantitative analyses in the paper are computed as full 32-layer sweeps (Figs. 4–6 are curves over all layers); the selected layers are used only for single-layer snapshots. Recomputing every headline quantity at each layer in a ±3 neighborhood of the peak (L9–L15): cos(over-refusal, harmful-refusal direction) stays within **0.395–0.458**, and the inter-task silhouette within **0.327–0.357** — narrow plateaus, so no conclusion depends on the exact layer choice. Your reading is exactly right, and we will add the full robustness table to the appendix.

> Relatedly, while stress-testing our own claims we found one that does *not* survive: the subspace-dimensionality comparison (§4.2, Fig. 6) is confounded by unequal sample sizes, and fails a matched-sample-size control in four independent conditions. We will revise or remove it in the revision (details in the General Response, point 3).

---

**Comments and suggestions:**

> **Early representative example:** agreed — we will add a paired example (one harmful-refusal, one over-refusal prompt) to Section 1, so both refusal types are concrete before the geometry begins.

> **Typos / missing articles:** thank you — we have done a full grammar pass for the revision.

> **Datasets and software:** the evaluation dataset is prepared for release, and we will release the complete notebook pipeline (dataset construction, embedding extraction, all analyses, and the new rebuttal experiments) with the camera-ready version.

---

We believe these additions — conversational task coverage, the layer-neighborhood verification, and the artifact release — address both weaknesses, and we thank you again for the constructive feedback.
