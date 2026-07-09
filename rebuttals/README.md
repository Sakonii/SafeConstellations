# EMNLP 2026 (ARR May) Rebuttal Package — Submission 12602

Rebuttal materials for *"Over-Refusal and Representation Subspaces"*.
Reviews (full text in `reviews_raw.md`): **xCj3 = 3.5** (conf 4, soundness 4), **Zici = 3.5**
(conf 2, soundness 4), **JwzU = 2** (conf 3, soundness 3 — "resubmit next cycle").

## Current status (2026-07-09)

**Response docs FINALIZED (pending user's read-through) for everything except JwzU W3.** All
six original notebooks run; all four response docs rewritten 2026-07-09 in the style of the
prior `safeconstellations.md` rebuttal: ~half the length, tables for numbers, plain direct
sentences. No placeholders remain (grep-verified). The former open brackets were resolved:
JwzU-W3 now says "a model with a non-trivial harmful-refusal rate" without naming it; xCj3-W3
suggests the title change inline. One number corrected during the rewrite: POS-tagging
over-refusals are **3**/205 (per the final judged CSV that R02 loaded), not the stale
pre-recovery "4" the earlier drafts quoted. Each doc keeps a single one-line HTML
identification comment at the top (invisible if pasted into OpenReview markdown; delete if
pasting elsewhere).

**New today: R07 drafted (Qwen cross-model directional replication, JwzU W3) — not yet run.**
Also: a real, previously-unfixed bug found and fixed in R01 (see below). See "R07" and "R01
Drive-persist bug" sections below.

### Over-refusal task coverage — a sanity check, not a bug (2026-07-09)
User asked: "it seems like we ain't seeing over-refusal in our tasks, is that correct?" —
confirmed: **5 of the 8 LLaMA tasks essentially never elicit OR** (cryptanalysis/rag_qa: 0
both in the original paper and here; pos_tagging/char_shuffle/word_shuffle: 3/3/1). Only
conversational_qa/translate/sentiment_analysis carry the OR population (75/18/13). This is
**not a problem** — it's already the headline finding, not a gap: over-refusal concentrates
in tasks requiring open-ended semantic engagement and is absent in mechanical transformations
of the same content, which is what makes the conversational result mean something rather than
being one more coin flip. **Decision: do not add more LLaMA OR-eliciting task types** —
diminishing returns given 3 solid tasks + a clean 5-task negative control; time is better
spent on the one concern still resting on a commitment rather than a result (JwzU W3, below).

### R07 — Qwen cross-model directional replication (drafted 2026-07-09, NOT yet run)
`R07_[Rebuttal]_Qwen_CrossModel_Directional_Replication.ipynb`. Targets JwzU W3 specifically:
Qwen1.5-7B-Chat (paper appendix) produced exactly 1 harmful refusal on the 270-prompt set,
leaving the harmful-refusal DIM direction undefined — the "central directional comparison"
JwzU asks about was never computable on that model. R07 reruns the paper's exact extraction
pipeline (hooks/judge verbatim from R01, `_get_input_ids` fix included) on
`Qwen/Qwen2.5-7B-Instruct` (config-swappable — stronger safety post-training than Qwen1.5, a
natural next model to try) to check whether it produces enough harmful refusals to unblock
the comparison. Design notes:
- Zero dependency on any LLaMA-specific file — pulls the exact same 270 prompts fresh from
  `Sakonii/OveralignDataset` (byte-identical to every other notebook here). `INCLUDE_NEW_TASKS`
  flag (off by default) can regenerate R01's 4 new-task prompts inline too, if a fuller
  cross-model "more tasks" story is wanted later — not required for W3 itself.
- `NUM_LAYERS` auto-detected from `model.config.num_hidden_layers` (not hardcoded), so
  swapping to a different Qwen/model size needs no other code change.
- Step 4's mask-count cell is a **go/no-go checkpoint**: if RH is still <5 on this model, every
  downstream analysis cell prints `SKIPPED` and says so rather than computing misleading
  numbers from too few samples — this notebook cannot silently produce a bad result.
- If RH≥5: computes cos(OR dir, RH dir) with a 2000-resample bootstrap CI (the number JwzU W3
  is actually asking for, computed natively on Qwen's own residual stream — no cross-model
  Arditi-vector transfer, which wouldn't be meaningful) plus the matched-n dimensionality test,
  for consistency with the rest of the package.
- Skips the Helbling harmfulness self-check (only the 3-class refusal judge is needed for
  this notebook's specific analyses) — halves the judging cost vs. R01/NB4.
- Validated (21 cells, `ast.parse`+`nbformat.validate` clean) and round-tripped identical
  against `build_r07.py`.

**Once run:** if `[R7.3]`/`[R7.5]` come back with the CI excluding 1.0 and a point estimate in
a similar band to LLaMA's, fold them into `reviewer_JwzU.md` W3 in place of the current
commitment sentence — exactly the same process used for R04 today. If RH is still <5, W3
stays a commitment and the note in Step 4 suggests trying a different/larger model.

### R01 Drive-persist bug — found still active today, now fixed for real
Earlier project memory claimed this was already fixed ("persist moved to end of Step 4"), but
a direct re-read of the current notebook file showed the fix had **not actually landed**: cell
`cell-010` still executed `drive.mount` + `cp -a` immediately after extraction, before judging
(cells 12–13) ran — the exact bug that caused the original `KeyError: 'refusal_class'` in R02.
(In practice this was being masked by the user manually re-running the Drive-copy a second
time after judging, and/or via the one-off `_RECOVERY_rejudge_r01_from_drive.ipynb` — the
notebook file itself was never structurally corrected.) Fixed now: `cell-010` only saves
locally and explicitly comments why it must not touch Drive yet; a new cell (`5e3609b1`)
inserted right after judging (`cell-013`) does the actual Drive persist. Also updated the
closing "rerun with `MODEL_NAME='Qwen/Qwen2.5-7B-Instruct'`" note (cell `cell-016`) to point at
R07 instead, since literally rerunning R01 in place would overwrite `rebuttal_embeddings/`
and clobber the LLaMA bank R02/R04/R05 depend on. Validated (18 cells) and round-tripped
identical against a regenerated `build_r01.py`.
**Note:** while investigating this, also confirmed the pos_tagging OR=4→3 discrepancy
(mentioned above) is explained by this same recovery history: R01's own displayed cell output
(a stale, pre-recovery run) shows OR=4; the Drive copy R02 actually loads reflects the
recovery notebook's re-judging, which flipped one borderline sample (OR=3). Both numbers are
real outputs from real runs of the same pipeline; **3 is correct** since it's what every
downstream notebook actually consumed.

### R04 outcomes (run 2026-07-09) — mixed; docs reframed honestly
- `[R4.6]` **content-blocked task probe = 0.982 ± 0.011** (8-way, GroupKFold by content id,
  chance 12.5%) — the star result; now the load-bearing content control in all three docs.
- `[R4.2]` template-variant silhouette = **+0.03** (near zero) — phrasing contributes nothing.
  But `[R4.1]` task silhouette among the 4 new frames is only 0.072 (thin frames over one
  shared content pool — consistent with R02's 8-task silhouette finding), so the draft's
  "cluster by task, not phrasing" framing was replaced by "phrasing ≈ 0 + probe = 98.2%".
- `[R4.5]` distance ratio = **0.78 (< 1!)** — same-content/cross-frame pairs are *closer* than
  different-content/within-frame pairs; the draft's claim "the frame moves a representation
  more than a complete change of content" was **false on this data** and removed. Honest
  reframe used instead: same-content/cross-frame pairs are lexically near-identical controlled
  prompts, and the probe still decodes task at 98.2% on held-out contents.
- `[R4.7]` XSTest minimal-pair probe = **VOID** — only 2 refused safe-twins exist in the bank
  (vs 18 unsafe); probe degenerate at 0.500 everywhere. Not quoted anywhere. All three docs now
  honestly state the sub-question can't be answered on current data, partially concede the
  early-layer lexical point, and commit to a powered XSTest analysis in the revision.

### R05 outcomes (run 2026-07-09) — unfavorable; NOT cited in any response doc
Only `conversational_qa` was steerable (OR≥5). Results: OR rate 100%→60% (task-steer) vs
→90% (arditi), but RH retention collapsed (75%→5% under task-steer) → selectivity **0.57
(arditi 0.18)** — ordering replicates (task > global) but **both < 1**, vs paper Table 3's
1.57 (>1) on the original tasks. Worse: CRR on unframed JailbreakBench harmful prompts drops
**95% → 15%** — the conversational steering vector acts as a de-facto global refusal ablation
(jailbreak). No τ-gating was applied (deployment config gates steering by task-alignment
threshold), so ungated numbers likely overstate deployed risk — but gated numbers were not
measured and cannot be claimed. **No response doc references [R5.x]** (R05 was always a
self-imposed strengthener, not a reviewer ask) — recommendation: exclude from the rebuttal,
treat as revision/camera-ready material. Mechanistically the result is *consistent* with the
paper's story: the conversational frame is the thinnest, its OR direction sits closest to the
global refusal direction (cf. R02), so steering with it degenerates toward global ablation —
a genuine scope limitation for the method as task frames approach bare conversation.

### R01 — "more tasks" story (done)
Ran with `INCLUDE_CONVERSATIONAL=True` and content pooled across all 4 benign tasks (205
contents, not 60). Clean split: `pos_tagging`/`word_shuffle`/`char_shuffle` elicit ~0%
over-refusal; `conversational_qa` elicits 37% (n=75/205) — larger than any task in the
original paper. Per-task OR rates form a clean dichotomy: semantic-engagement tasks
(conversational 37%, translation 30%, sentiment 24%) vs. structural/mechanical tasks
(shuffle/tag/decode, 0–1.5%). Gives a genuine 3rd OR task (sentiment_analysis, translate,
conversational_qa) for per-task directional analysis.

### R02 — geometric re-tests (done, including the dimensionality investigation)
- `[R2.1]` 8-task silhouette = 0.121 (vs paper's 0.454) — explained, not a regression: 4 new
  tasks share one content pool with thin task-frame wrapping, so they cluster near each other
  while each still separates from the original 5 (confirmed via galaxy plot).
- `[R2.4]`/`[R2.5]`/`[R2.7]`/`[R2.8]`: 3 valid OR tasks now (was 2); per-task OR cosine
  0.428±0.058 vs. a **freshly recomputed** RH-alignment anchor `[R2.17]`=0.536±0.086 (not the
  paper's old 0.845 — see "data-provenance" note below). OR remains the more task-divergent of
  the two. cos(OR,Arditi)=+0.464, consistent with the original +0.448.
- **Dimensionality claim (§4.2: OR needs more PCs than RH) — RESOLVED, retracted.** Tested
  under 4 independent matched-n conditions: 5-task original (0.90%), 8-task pooled/dominated
  by conversational_qa (0.00%), task-balanced (0.50%), and — the cleanest test, excluding
  conversational_qa entirely so no task dominates (0.00%). All four land in the same ~0–1%
  range; participation ratio never favors OR either. The dominance-artifact hypothesis (that
  conversational_qa's 66% share was masking a real effect) does **not** hold up — even the
  undominated 5-task population shows no gap. **All four response docs now state this claim
  will be revised/removed in the camera-ready version**, framed as a demonstration of rigor
  rather than a weakness (this is exactly what JwzU asked for).

### R03 / R06 (done, unaffected by the above)
R03: bootstrap CIs, permutation tests, layer-neighborhood robustness — all filled. R06: Claude
second judge, 95.2%/97.6% agreement with GPT-4o; 22% classifier-refusal rate reported as an
honest aside; no human-annotated validation performed (explicitly not claimed — planned for
camera-ready).

**R03 stale-reference bug (found & fixed during the consistency sweep):** R03's `[R3.1]`/`[R3.2]`
cell and its markdown originally compared cos(OR, Arditi)'s bootstrap CI against the same old
"0.835–0.858 harmful-refusal alignment" reference that turned out to be data-provenance-stale
(see note below) — never updated when R02 moved to the fresh-recompute policy. Once the honest
fresh reference (`[R2.17]`=0.536±0.086) is used instead, it actually **overlaps** with R3.1's CI
`[+0.349, +0.530]`, so the original "clearly separated from the harmful-refusal band" framing was
no longer supportable. Fixed by dropping the external-reference comparison entirely and keeping
only the claim that doesn't need one: the CI excludes 1.0 (perfect alignment) by a wide margin.
Applied to the R03 notebook (cells `cell-002`/`cell-003`/`cell-004`/`cell-012`) and propagated
the same wording softening to `00_global_response.md` §3 and `reviewer_JwzU.md` W1 (both already
"finalized" at the time — user approved the change before it touched those docs).

### Data-provenance note (resolved pragmatically, not fully root-caused)
The `./embeddings` bank's per-task totals don't match the paper's own Appendix Table (e.g.
sentiment_analysis: 55 here vs. 60 in the paper — a subset can't exceed its superset, so this
is a genuinely different snapshot, not a subsampling artifact; CSV timestamp `20250714`
suggests an old, unsynced extraction). Confirmed not a wrong-file-picked bug (only one CSV
found per directory). **Resolution:** stopped quoting old hardcoded paper reference numbers
(0.845 alignment, 11-vs-8 dimensionality) in new analyses; R02 now recomputes references fresh
from the current bank (`[R2.17]`). This is noted here for the record but was **not** exposed to
reviewers in the rebuttal text itself — it's an internal data-hygiene note, not something
reviewers asked about, and surfacing it could invite unnecessary scrutiny of data integrity.
**Action item before camera-ready:** reconcile which bank is canonical and ensure the
paper's own numbers are internally consistent.

### Bugs found & fixed along the way
1. **R01→Drive staleness** (`KeyError: 'refusal_class'`): R01's Drive-persist ran before
   judging completed. Recovered via `_RECOVERY_rejudge_r01_from_drive.ipynb` at the time (no
   GPU needed, judging is CPU+API only) — but the notebook's actual cell ordering was **not**
   structurally fixed until 2026-07-09 (see "R01 Drive-persist bug" above); an earlier version
   of this README/memory claimed it was fixed when it wasn't. It is now.
2. **R02 `BENIGN_TASKS` staleness**: whitelist predated `conversational_qa`, silently zeroing
   its entire OR population (75 samples) out of every OR analysis while `RH` (independent of
   the whitelist) came through fine. Fixed — confirmed working on the corrected re-run.
3. `load_bank()` in R02/R04/R05 checks for required columns upfront with an actionable error
   (guards against repeats of bug #1) and lists every CSV found per directory (guards against
   picking the wrong file if a duplicate ever appears).
4. **R04 missing Colab-load cell**: unlike R02/R05/R06, R04 had no drive-mount/copy step at
   all — would fail immediately on a fresh Colab runtime. Added the standard commented block
   (mirrors R02/R05).
5. **R04 content-matching bug (genuine correctness bug, not just missing infra)**:
   `extract_content()` was a single generic regex tuned for `sentiment_analysis`/`translate`/
   `rephrase`'s tail-quote pattern — it silently failed (returned `None`) on `cryptanalysis` and
   `rag_qa`, which R01 actually draws pooled content from, and it referenced `rephrase`, which
   R01 never draws from at all. This would have quietly shrunk the content-matched control
   populations (Control 2/3: same-content-different-task, content-blocked task probe) without
   raising an error. Fixed by mirroring R01's exact per-task extraction logic
   (`ORIGINAL_CONTENT_TASKS = ['sentiment_analysis', 'translate', 'cryptanalysis', 'rag_qa']`,
   task-specific regexes for cryptanalysis/rag_qa) and adding a per-task match-count diagnostic
   print so a silent drop like this would show up immediately next time. Also strengthened
   Control 3 (content-blocked task probe) to use all matched original tasks instead of only
   `sentiment_analysis`. Validated (11 cells, `ast.parse` + `nbformat.validate` clean) and
   round-tripped identical against a regenerated build script.
6. **R05 audit (no bugs found)**: confirmed R05 already had the Colab-load cell (commented,
   consistent with R02/R06), no `BENIGN_TASKS`-style hardcoded whitelist, and dynamic task
   handling (`NEW_TASKS = sorted(np.unique(tasks_r))`). One pre-existing, non-bug design choice
   noted: the CRR (correct-refusal-rate) safety population steers with
   `STEERABLE_TASKS[0]` (alphabetically first steerable task) since CRR prompts have no task
   frame of their own to key off of — a reasonable, deliberate choice, left as is.

## Review → response → experiment map

| Reviewer concern | Response section | Evidence notebook |
|---|---|---|
| JwzU W1 + comment: OR in only 2 tasks, "evaluate more tasks where over-refusal actually occurs" | JwzU-W1, global §1 | **R01 + R02** ✅ done |
| JwzU W1: small dataset / statistical validation | JwzU-W1, global §3 | **R03** ✅ done |
| JwzU W1 (implicitly): dimensionality claim rigor | JwzU-W1, Zici-W2, global §3 | **R02** ✅ done — **claim retracted** |
| JwzU W2 / xCj3 sugg.: template & lexical confounds, controlled prompt pairs | JwzU-W2, global §2 | **R04** ✅ done — probe 98.2%; [R4.5]/[R4.7] reframed honestly |
| JwzU W3: partial cross-model replication | JwzU-W3 | commitment; **R07** drafted 2026-07-09 for Qwen2.5-7B-Instruct, not yet run |
| xCj3 W1: judge validation (second judge, different model family) | xCj3-W1, global §4 | **R06** ✅ done |
| xCj3 W2: causal patching = negative evidence only | xCj3-W2 | wording commitment (+20 pairs, camera-ready) |
| xCj3 W3: scope abstract/title claims | xCj3-W3 | wording commitment |
| Zici W1: conversational-agent relevance | Zici-W1 | **R01** ✅ done — OR=75/205 (37%) |
| Zici W2: layer-selection robustness | Zici-W2 | **R03** ✅ done |
| Zici/JwzU: datasets & software = 1 | Zici comments, global §5 | release commitment |

## What's left

```
1. R07 — GPU + OpenAI judge, no dependency on other notebooks — fills [R7.x]
```
R05's results are deliberately not cited in any response doc (see "R05 outcomes" above). If
R07's directional comparison comes back usable, fold `[R7.3]`/`[R7.5]` into `reviewer_JwzU.md`
W3 (same process as R04 → JwzU-W2 today); otherwise W3 stays a commitment and no doc changes.

R01–R06 assume the NB14 Colab layout: original memory bank in `./embeddings/`
(from Drive `embeddings/overalign_eval/llama`), NB8 artefacts in `./arditi_artefacts/`.
R01 writes `./rebuttal_embeddings/`, which R02/R04/R05 read. **R07 is independent of all of
this** — it pulls prompts directly from `Sakonii/OveralignDataset` and writes its own
`./qwen_crossmodel_embeddings/`.

## Placeholder convention

Each notebook ends with a **CONSOLIDATED REBUTTAL NUMBERS** cell printing tagged values
(`[R2.5] …`). The four response markdowns use the same tags. Notebooks also print
**interpretation guides** ("if the number looks like A, argue X; if B, argue Y"), mirrored by
`<!-- -->` comments in the markdowns — resolve those branches based on the actual numbers,
then delete the comments before posting.

## Pre-posting checklist

- [x] R01, R02, R03, R06 run and their numbers filled into all four response docs
- [x] Dimensionality claim finalized (retracted, framed as rigor) across global §3, JwzU-W1, Zici-W2
- [x] R04/R05 run; `[R4.x]` filled (with honest reframes), R05 deliberately excluded
- [ ] Run R07 (optional but recommended — the only remaining reviewer concern still resting
      on a commitment rather than a result) → fold `[R7.x]` into JwzU-W3 if RH≥5
- [ ] Delete all HTML comments before posting
- [x] No numeric placeholders remain (grep-verified; only the title-suggestion and second-model-name brackets are left, both deliberate)
- [ ] Check ARR response length limits (global response + per-reviewer replies)
- [ ] Decide on xCj3-W3 title question (keep or narrow the title — response currently asks for the reviewer's preference)
- [ ] Reconcile the original-bank data-provenance issue before camera-ready (see note above) —
      not reviewer-facing, but needed for internal consistency
- [ ] Post: global response first, then per-reviewer replies (JwzU first)
