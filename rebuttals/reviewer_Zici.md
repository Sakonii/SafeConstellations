<!-- REVIEWER Zici: Overall 3.5, Confidence 2, Soundness 4. -->

**Response to Reviewer Zici**


Thank you for the positive assessment and the thoughtful potential weaknesses. Below we discuss the weaknesses and attempt to address them with a few new experiments.

---

**W1: The five task domains are not very compelling for this question; over-refusal matters most for conversational systems.**

> **Response:** While conversational way of interacting with LLMs could vary from conversation to conversation and is difficult to analyse mechanistically, we experiment an additional task where we add **conversational framing**, ("Reply helpfully to this user message: …<text>"). 
(Additionally for comparison, we also extended three more tasks on POS tagging, word/character shuffling during the review period).

> | Task frame | Refusal rate |
> |---|---|
> | Conversational | **37%** (more than any task) |
> | Translation / sentiment analysis | 30% / 24% |
> | POS tagging / character shuffling / word shuffling | 1.5% / 1.5% / 0.5% |


> The closer a task frame comes to having an open-ended conversation (say responding to user’s harmful instructions), the more refusal occurs. And tasks that deviate from responding to text (sentiment, shuffling of words, tagging, etc) refuse less, despite wrapping identical text content. We believe conversation-like framing is similar to having no tasks for single-turn conversations. Mechanistically analysing multi-turn conversation (to-and-fro between users and models) requires analysis on multiple tokens, which is currently beyond the scope of this work. We will make this conversational frame a full task in the camera-ready and reframe the discussion (§3.1) around this.

---

**W2: The layer selections are well motivated but the differences from neighboring layers look minor; would results differ in a small neighborhood around the maximal points?**

> **Response:** We appreciate the reviewer's careful observation. All quantitative analyses are computed as full 32-layer sweeps (Figs. 4-6 are curves over all layers); and the selected layers are used only as a candidate only for single-layer snapshots and for further analysis. By recomputing every headline quantity at each layer in a ±3 neighborhood of the peak (L9 to L15): cos(over-refusal, harmful-refusal direction) stays within **0.395-0.458**, and the inter-task silhouette within **0.327-0.357**. Both are narrow plateaus, and no conclusion depends on the exact layer choice. We will add this additional discussion and full robustness table to the appendix.


---

**Regarding Comments and suggestions:**

> **Early representative example and Typos:** As suggested by the reviewer, we will add examples (one harmful-refusal, one over-refusal prompt) around earlier sections to improve clarity of writing.

---

We believe these discussions (conversational task, the layer-neighborhood discussion) acknowledge some of the weaknesses raised, and we thank the reviewer again for the constructive feedback. All discussions will be incorporated in the camera-ready version.
