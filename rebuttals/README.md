# EMNLP 2026 (ARR May) Rebuttal Package — Submission 12602

Rebuttal materials for *"Over-Refusal and Representation Subspaces"*.
Reviews (full text in `reviews_raw.md`): **xCj3 = 3.5** (conf 4, soundness 4), **Zici = 3.5**
(conf 2, soundness 4), **JwzU = 2** (conf 3, soundness 3 — "resubmit next cycle").

## Current status (2026-07-13)

**Overall official comment added (`00_overall_official_comment.md`), amending the 2026-07-11
decision.** The three per-reviewer replies remain the only place with technical content; what
changed is that a short top-level official comment (visible to all reviewers and the AC) will
now also be posted: scores summary, shared and per-reviewer strengths quoted from the reviews,
thanks, a one-sentence list of the response-period experiments, and the release commitment.
No numbers, no CIs, no dimensionality mention, zero em-dashes (grep-verified). ~280 words.
Posting order: overall comment first, then per-reviewer replies (JwzU first).

**R12 run: XSTest lexical-confound check (JwzU W2 point 3).** Real result: Layer 1 balanced
accuracy for xstest_safe vs xstest_unsafe is exact chance (0.500), crossing 80% by Layer 2,
95% by Layer 4 — contrasts with the paper's own §4.4 refusal-type probe (100% from Layer 1,
different, unmatched population). Caveat disclosed in the doc rather than hidden: a plain
bag-of-words baseline on the same text gets 98.2%, so the XSTest pool isn't as lexically
matched in practice as intended; the finding is "not decodable from L1" rather than "not
lexical." Folded into `reviewer_JwzU.md` W2 point (3) with a commitment to soften the paper's
"100% from Layer 1" wording in the revision. Also fixed a wrong section reference (was §4.3,
is actually §4.4 per `live.main.tex`'s subsection order).

## Current status (2026-07-11, final restructure)

**JwzU letter rewritten for score impact (user directive):** R11's strict task-wise rates
promoted into the W1 headline table (second column); old W2 point (4) deleted (one-sentence
pointer remains). **Dimensionality retraction removed from all reviewer-facing letters** —
the retraction stands internally and §4.2/Fig. 6 must still be revised in the camera-ready;
it is simply no longer volunteered in the rebuttal. All 95% CIs removed (plain wording backed
by the same bootstraps). JwzU ~980 words, Zici ~455, xCj3 ~354.

## Current status (2026-07-11, updated)

**R11 run: task-wise over-refusal redefinition (JwzU-W2 point 4, Zici-W1 caveat).** The
earlier check (R09 Analysis 0) used one shared "concerning content" rule across continuation,
draft_message, and conversational_qa, and reported a share-of-refusals (169/318), not a rate.
`R11_[Rebuttal]_CaseByCase_Over_Refusal_Redefinition.ipynb` redefines which content types
count as genuine over-refusal **per task**: a written-out harmful response is risky to
continue/draft around but not obviously risky to reply to; a harmful prompt/instruction is
the reverse (replying can mean answering it, continuing its wording doesn't). Real corrected
rates (refusals over the correct eligible-content denominator): continuation 37.9% (162/427),
conversational_qa 33.3% (93/279), draft_message 25.8% (110/427, assumed same rule as
continuation, user has not explicitly confirmed this one). All three tasks stay valid for the
six-task directional analysis (6/6 unchanged). Both docs updated with the real rates and an
explicit mention of the per-task redefinition.

**Known nuance, not written into any doc:** R11 also recomputed the six-task OR-direction
pairwise cosine under the corrected population: 0.358±0.235 (original) -> 0.246±0.336
(corrected) — mean drops (arguably a stronger task-specificity signal) but the standard
deviation grows past the mean, a much noisier estimate, since continuation/conversational_qa's
genuine-OR sample counts shrank a lot (306->162, 318->93). **Decision (user, 2026-07-11):
footnotes only** — leave JwzU-W1's headline "0.358 ± 0.235" as-is; do not fold the noisier
corrected number into the headline.

**Bug found in R11, not a real finding:** its "global OR direction vs Arditi" analysis
(printed as [R11.7]/[R11.8], +0.555 -> +0.522) used `HARMLESS_ANSWERED` as the baseline
instead of the project's established `TARGET_MASK` (a Helbling-judge-filtered baseline,
original+R01 rows only) that R09/R02 use for this specific quantity — these two numbers do
not match anything already published (+0.410 to +0.464 range) and were NOT used anywhere.
Only R11's per-task analysis (which does correctly use per-task HA, matching R09/R02) and the
Analysis-0 rate table were used.

## Current status (2026-07-11)

**Structural decision: no general response will be posted.** Reviewers don't see each
other's threads, and a shared response reads as untailored. Every point that used to live
only in `00_global_response.md` has been folded directly into the relevant per-reviewer
doc, so each of the three (`reviewer_JwzU.md`, `reviewer_xCj3.md`, `reviewer_Zici.md`) is
now fully self-contained. The former global response is kept as
`_NOT_POSTED_global_response_draft.md`, an internal reference only, not part of the
posting plan. The user has also taken direct editorial control of `reviewer_xCj3.md` (hand
edited 2026-07-11); treat that file's prose as user-owned going forward, and always
re-read it before editing rather than assuming the last-known version still matches.

**R07, R08, R09, R10 ALL RUN. All three response docs fully updated with the final,
full-dataset numbers, including a resolved methodological question the user raised about
the OR definition itself, and a genuine tightening/clarity pass. Package is essentially
complete pending the user's own final read-through.**

**R08** (user-modified before running: 3 tasks, single template each, 615 prompts on the
205-content pool; `advice_seeking`/`open_qa` dropped, `summarize` replaced by
`keywords_identification`) — all three cleared the Stage-2 bar decisively (pilot-scale:
continuation 78/205=38%, beats conversational_qa's 37%; draft_message 47/205=23%;
keywords_identification 39/205=19%). `keywords_identification` is a nice mechanistic bonus:
a short labeling task (like sentiment, 24%) that still triggers OR, vs POS tagging (also
labeling, but syntactic) at ~1.5% — sharpens "semantic engagement vs structural
transformation" over "long vs short output."

**R10** (run 2026-07-10): expanded continuation/draft_message/keywords_identification/
conversational_qa onto ~505 net-new train-split contents, 2,020 prompts. Rates held or rose
at scale on the full merged bank (test-pool + train-pool combined, ~645 prompts per task):
**continuation 306/644=47.5% (rounds to 48%), conversational_qa 318/645=49.3%, draft_message
184/643=28.6%, keywords_identification 155/645=24.0%** — all higher than the pilot rates,
confirming no dilution effect from the larger, more diverse content pool. (Caught and fixed
a rounding error in the docs: continuation's full-scale rate was mistakenly written as 47%
in two places; 47.52% correctly rounds to 48%.)

**R09** (re-run after R10, full 4-bank merge, 3,725 samples, 12 tasks) hit the decision
guide's "use extended numbers" branch: **[R9.2] 6 valid OR tasks** (was 3), **[R9.3] OR
pairwise cosine 0.358±0.235** vs **[R9.4] fresh RH alignment 0.636±0.140 (N=7)** — wider
contrast than R02's 0.428-vs-0.536. [R9.6] cos(global OR, Arditi)=+0.410 (stable vs
+0.464/+0.448). [R9.8] dimensionality 0.00% (consistent with the retraction).

**R07 (run 2026-07-10) — JwzU W3 is now a genuine result, not a commitment.** Qwen2.5-7B-
Instruct, all 7 rebuttal frames + originals (1,705 prompts): **OR=270, RH=73**, both far
above the n=5 minimum (Qwen1.5-7B reference: OR=22, RH=1, undefined direction). Results,
reported with full honesty rather than cherry-picked:
- **cos(OR dir, RH dir) = +0.784, 95% CI [+0.724, +0.823]** — excludes 1.0, so directions
  remain distinct, but notably MORE aligned than on LLaMA (+0.464, CI [+0.349,+0.530]).
  Framed in the docs as a **partial** replication: the qualitative claim holds, the exact
  degree of separation is model-specific — not oversold as an exact match.
- **Task-conditioned pattern replicates cleanly**: mechanical tasks 0-4 OR/205, generative
  tasks 41-77/205 on Qwen2.5, matching the LLaMA split almost exactly (e.g. continuation
  77/205=37.6% on Qwen2.5 vs 78/205=38% on LLaMA). This is a genuine bonus result extending
  the "more tasks" story (JwzU W1/Zici W1) to a second model family, added to global point 6.
- **Dimensionality flips**: the matched-n test that retracted the dimensionality claim on
  LLaMA (<0.9%) gives **86.8%** on Qwen2.5 — i.e., the SAME rigorous test that killed the
  claim on LLaMA finds it holds on Qwen2.5. Reported honestly as a "model-dependence" nuance
  in JwzU-W3, explicitly NOT reopening the LLaMA-specific retraction (which stands as-is
  everywhere else in the docs) — this is delicate wording, worth the user double-checking.
- Silhouette (0.131 @ L18, vs LLaMA 0.454/Qwen1.5-paper 0.458) is lower, consistent with the
  already-explained "many thin task-wrappers over one shared content pool" pattern seen in
  R02/R09 on LLaMA too — not a new concern, and not mentioned in the docs (kept them focused
  on the directional comparison JwzU actually asked for).
- R07 notebook: user added a parallelized (ThreadPoolExecutor) judging cell in Colab,
  replacing the sequential one — left as executed, not rewritten, since it ran successfully;
  only the trailing empty Colab cell was cleaned up.

**Doc tightening pass (2026-07-10, per user request for "clear, easy to follow, concise"):**
merged the separate pilot-scale and full-scale over-refusal tables/paragraphs in
`00_global_response.md` §1 and `reviewer_JwzU.md` W1 into single two-column tables (was:
table + a near-duplicate paragraph repeating the same numbers); trimmed W2 point (4) and W3
for redundant phrasing; disambiguated "per-task pairwise OR-direction cosine" from "global
OR-vs-Arditi cosine" in W1 (two different +0/±0 numbers appeared back to back and could read
as contradictory). Word counts after tightening: global 1,064 (was ~1,074), JwzU 1,232 (was
1,326), xCj3 589, Zici 572. **Fixed a stale reference**: xCj3-W3's scoping sentence still
said "partial replication on Qwen1.5-7B... eight NLP task frames" — updated to name the
completed Qwen2.5 cross-model replication and the correct task count (twelve: 5 original + 7
new).

### Methodological question raised by the user (2026-07-10) — investigated and resolved
User asked: for tasks like `continuation`, doesn't refusing to continue a `harmful_response`
(e.g. a written-out keylogger script) count as *correct* refusal, not over-refusal, since
compliance would mean generating more harmful content? Whereas `keywords_identification` is
benign no matter the input. **Verified this is a real, well-founded distinction** — and not
new to R08/R10: the paper's own original masking already treats `harmful_response`/
`jailbreak_prompt`/`xstest_unsafe` content as OR-eligible for sentiment/translate, and
`conversational_qa`'s already-published figure inherits the same construction.

Added a diagnostic (R09 Analysis 0, cell `4f76cae4`) breaking down each generative task's OR
by content type. Findings, on the full merged bank:

| Task | Total OR | Concerning (harmful_response+jailbreak+xstest_unsafe) | Clean OR |
|---|---|---|---|
| continuation | 306 | 129 (42%) | **177 (58%)** |
| conversational_qa | 318 | 149 (47%) | **169 (53%)** |
| draft_message | 184 | 140 (76%) | **44 (24%)** |

("Clean" = excludes any content type where compliance could itself be harmful.
`decrypt_harmful_response`/`decrypt_harmful_instruction` — cryptanalysis-sourced ciphertext —
are kept in the clean bucket: continuing/drafting around scrambled ciphertext cannot itself
produce harmful output regardless of what it decrypts to, since these tasks don't ask the
model to decrypt anything.)

**Decision:** keep the headline OR counts/rates as originally defined (consistent with the
paper's pre-existing methodology; do not retroactively redefine `conversational_qa`'s
published figure). Report the clean-subset numbers as a proactive robustness check instead —
**added as JwzU-W2 point (4)**, plus a one-sentence mention in global §1/§2. The effect
survives even under the strictest reading for all three tasks; `draft_message` is honestly
the weakest (24% clean, n=44, still >> the n=5 minimum).

**Docs updated with all of the above:** global §1 (scale-up + robustness sentence), JwzU-W1
(7-row table description + full-scale confirmatory numbers + updated 6-task directional
figures), JwzU-W2 new point (4), Zici-W1 (one sentence on the scale-up). All four docs
re-verified: zero em-dashes, no stale 3-bank/pilot-only numbers left uncontextualized.

**R09 notebook updated to match:** header table now shows the final 4-bank numbers (not the
3-bank pre-R10 ones); Analysis 0's markdown documents the resolution above. Validated
(15 cells) and round-tripped clean.

**Also fixed:** R07's `INCLUDE_NEW_TASKS` now covers ALL 7 rebuttal frames (R01's 4 with
3-template round-robin + R08's 3 with single templates, matching the LLaMA banks exactly)
and defaults to True (1,705 prompts, ~95 min extraction + ~32 min judging; False = minimal
270-prompt W3-only run) — de-risks the OR side of the W3 go/no-go, since Qwen1.5's OR on the
original 270 was only 22 samples. R07's `BENIGN_TASKS` is now built dynamically.

**Remaining:** none of the planned experiments. Just the user's own final read-through,
deciding the xCj3-W3 title question, deleting HTML comments, checking ARR length limits
(JwzU is the longest at 1,232 words), and posting (global response first, as multiple
follow-up comments if needed, then per-reviewer replies with JwzU first).

---

## Previous status (2026-07-09)

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

**Also new today: R08 drafted (High-OR task pilot, JwzU W1) — not yet run. Stage 0 (free,
already run) found nothing actionable.** See "R08" section below.

### Over-refusal task coverage — a sanity check, not a bug (2026-07-09)
User asked: "it seems like we ain't seeing over-refusal in our tasks, is that correct?" —
confirmed: **5 of the 8 LLaMA tasks essentially never elicit OR** (cryptanalysis/rag_qa: 0
both in the original paper and here; pos_tagging/char_shuffle/word_shuffle: 3/3/1). Only
conversational_qa/translate/sentiment_analysis carry the OR population (75/18/13). This is
**not a problem** — it's already the headline finding, not a gap: over-refusal concentrates
in tasks requiring open-ended semantic engagement and is absent in mechanical transformations
of the same content, which is what makes the conversational result mean something rather than
being one more coin flip. **Initial decision that day: do not add more LLaMA OR-eliciting task
types** — diminishing returns given 3 solid tasks + a clean 5-task negative control.

**Revisited later the same day:** user asked to reconsider, given that the pipeline was only
ever using a 270-row `test` split — the dataset's `train` split (777 rows, never touched by
any notebook) turned out to hold a ~3.5x larger, currently-unused content pool. See "R08"
below for the resulting staged experiment design.

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

### R08 — High-OR task pilot, Stage 1 of 2 (drafted 2026-07-09, NOT yet run)

Targets JwzU W1's explicit comment ("evaluate more tasks where over-refusal actually occurs")
with a fuller, cheaper-first approach than R01's single-shot 4-task try.

**Dataset investigation (via 2 parallel Explore agents + 1 direct verification), full facts
in `/Users/sakonii/.claude/plans/hidden-leaping-abelson.md`:** `Sakonii/OveralignDataset` has
a `train` split (777 rows) no notebook has ever used — everyone only loads `test` (270 rows).
Verified directly (`1. [Dataset] (train + test).ipynb`): `data_train =
pd.concat([data, data_test, data_test]).drop_duplicates(keep=False)` — train/test are disjoint
carve-outs of one shared pool. Train per-task counts: sentiment_analysis=197, translate=216,
cryptanalysis=186, rephrase=178, **rag_qa=0** (train has zero rag_qa content — all 30 rag_qa
rows live in test only). Deduplicated, train contributes 505 net-new contents beyond R01's
205-content pool → **test∪train = 710 unique contents**, a ~3.5x expansion, *if* we ever need
it (see Stage 2 below — we don't yet).

**Stage 0 (free, CPU-only, already run against local `embeddings_llama.csv` — no Colab
needed):** re-checked whether `rephrase` (excluded from all OR analysis as "can itself
constitute a harmful instruction") secretly hides genuine over-refusal on its benign-content
subset. Of the 270-sample bank's 65 rephrase rows, 19 are on genuinely benign content
(`benign_instruction`=7, `xstest_safe`=12); of those, **18 direct_answer, 1
indirect_refusal, 0 direct_refusal**. **Result: essentially nothing actionable** — 1 sample is
far below the n≥5 minimum for any direction estimate. Correctly excluded harmful/adversarial
rephrase rows (n=46) reconcile exactly against R02's already-known rephrase counts (24
direct_refusal + 1 indirect_refusal there, +1 indirect_refusal here = 2 total, matching).
**No change to any finalized doc** — this finding is too weak to justify reopening the
rephrase-exclusion framing in `reviewer_JwzU.md`/`00_global_response.md`, but it's worth
keeping in mind for Stage 2 (train has ~60 more untested benign-content rephrase rows).

**Stage 1 (R08 itself, not yet run):**
`R08_[Rebuttal]_HighOR_Task_Pilot.ipynb`. Pilots 5 new, mechanistically-motivated task types
on the SAME EXISTING 205-content pool R01 used (deliberately **zero new data pooling** in
this stage; regenerated inline from `Sakonii/OveralignDataset`, not dependent on any
R01-persisted file, mirrors R07's self-contained design):
- `advice_seeking`, `open_qa`, `summarize` (original 3 — see rationale above)
- `continuation` and `draft_message`, added after user pushback on 2026-07-09 wanting tasks
  that involve genuine free-text generation rather than narrow/labeling ones. Refined working
  hypothesis in the process: it isn't "generates lots of text vs. narrow" per se —
  `sentiment_analysis` is just as narrow/one-word-output as `pos_tagging`, yet gets 24% OR vs.
  ~1.5% — the better predictor is whether answering requires *semantic/evaluative engagement*
  with meaning (judgment, stance, new content) vs. mere *structural transformation*.
  `continuation` (extend the text) is rated the single highest-confidence bet to beat
  conversational_qa's 37%, since "produce more of it" is a stronger act than "reply to it".
  `draft_message` is a weaker OR bet but adds deployment-realism value.

Reuses R01's exact, already-fixed helpers verbatim (`_get_input_ids`, extraction-hook
pattern, `classify_refusal`); judges with `refusal_class` only (skips Helbling, same
cost-saving choice as R07). Writes to its own `rebuttal_embeddings_v2/` directory — never
touches R01's `rebuttal_embeddings/`. Scale: 5 tasks × 205 contents = 1025 prompts ≈ 76
minutes total (57 min extraction + 19 min judging, using R01's measured 3.34s/prompt and
1.13s/call rates). Validated (15 cells, `ast.parse`+`nbformat.validate` clean) and
round-tripped identical against `build_r08.py`.

**Go/no-go for Stage 2:** a task advances if OR n≥5, ideally ≥15–20 (well above R01's ~0–3
mechanical-task noise floor). **Stage 2 (a not-yet-built `R10` — renamed from "R09" when R09
became the merge notebook below) is conditional** — only built
for whichever task(s) pass, re-running just those on the expanded test∪train pool (up to 710
contents) for a much larger, better-powered OR population. Full staged design, cost formulas,
and the rejected "one-shot maximal" alternative are in the plan file referenced above.

**Once Stage 1 is run:** report headline OR/RH/HA counts per task; if ≥1 task clears the bar,
fold results into `reviewer_JwzU.md` W1 the same way R04's numbers were folded into JwzU-W2 —
report honestly regardless of outcome, including if all 5 disappoint (a fast, ~76-minute
negative result, consistent with R05's precedent).

### R09 — Extended Task Geometry, 3-bank merge (drafted 2026-07-09, NOT yet run; CPU-only)
`R09_[Rebuttal]_Extended_Task_Geometry.ipynb`. Answers "do we need to redo the other
notebooks after R08?" — mostly no: R03/R06 run on the original bank only, R04's headline
(98.2% probe) already lands, R05 is excluded. **R02 is the one analysis whose headline
numbers improve with more OR tasks**, so R09 recomputes R02's key quantities on the 3-bank
merge (original + R01 + R08, up to 2,115 samples / 14 task frames) **without touching R02**
— both number sets are kept and compared side by side, per the user's "keep track of the
current one, choose whichever works better" instruction. Design notes:
- `load_bank()` tolerates a missing `llm_evaluation` column (R08 judges refusal-only);
  TARGET (global-OR reference population) draws from original+R01 rows only, all per-task
  directions use HA which spans all three banks.
- `BENIGN_TASKS` built dynamically as `[t for t in ALL_TASKS if t != 'rephrase']` — immune
  to the R02 whitelist-staleness bug class by construction.
- Computes: merged silhouette `[R9.1]`, valid OR task count `[R9.2]` (vs R02's 3), per-task
  OR pairwise cosine `[R9.3]` (vs 0.428±0.058), fresh same-bank RH alignment `[R9.4]` (vs
  0.536±0.086), per-task + global cos vs Arditi `[R9.5]`/`[R9.6]` (vs 0.596/+0.464), and a
  matched-n dimensionality consistency check `[R9.8]` (expected ~0–1%, claim already
  retracted).
- The consolidated cell prints an explicit **decision guide with an honesty rule**: if the
  extended set strengthens the same conclusion, quote it in the docs in place of the 3-task
  version; if it weakens/reverses the OR-vs-RH task-specificity contrast, do NOT silently
  keep quoting the old number — surface it and choose the framing honestly (scope or report
  both), like the dimensionality retraction.
- Validated (13 cells) and round-tripped clean against `build_r09.py`. Runs in the same
  Colab session right after R08 (~10–20 min, CPU; Colab cell copies the other banks +
  arditi artefacts from Drive).

### Response-doc style pass (2026-07-09, per user request)
All four response docs rewritten to remove **every em-dash** ("looks too much AI written",
user's words) and match the plainer voice of the user's prior `safeconstellations.md`
rebuttal: simple sentences, commas/parentheses/colons instead of dash constructions, tables
kept, numeric-range en-dashes kept (they appear in the user's own writing). Verified by grep:
zero "—" characters remain in any of the four docs; word counts essentially unchanged
(global 745, JwzU 818, xCj3 583, Zici 525). All numbers untouched.

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

**Also fixed (cosmetic, found by a later investigation agent, verified directly):** R01's
last cell (`cell-016`) had a stray literal `</cell id="cell-016">` string accidentally copied
into its markdown source during the edit above — a Read-tool display-wrapper artifact
mistakenly included in a `NotebookEdit` call's replacement text. Harmless (renders as visible
text in a markdown cell, doesn't break execution) but sloppy; removed. Round-tripped clean
against a regenerated `build_r01.py` (18 cells, 0 mismatches).

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
| JwzU W1 + comment: OR in only 2 tasks, "evaluate more tasks where over-refusal actually occurs" | JwzU-W1 | **R01 + R02 + R08 + R10** ✅ done — 6 valid OR tasks now, full dataset |
| JwzU W1: small dataset / statistical validation | JwzU-W1 | **R03** ✅ done |
| JwzU W1 (implicitly): dimensionality claim rigor | JwzU-W1, Zici-W2 | **R02** ✅ done — **claim retracted on LLaMA** |
| JwzU W2 / xCj3 sugg.: template & lexical confounds, controlled prompt pairs | JwzU-W2, xCj3 | **R04** ✅ done — probe 98.2%; [R4.5]/[R4.7] reframed honestly; new point (4) OR-definition robustness check |
| JwzU W3: partial cross-model replication | JwzU-W3 | **R07** ✅ done — Qwen2.5, cos(OR,RH)=+0.784, CI excludes 1.0 |
| xCj3 W1: judge validation (second judge, different model family) | xCj3-W1 | **R06** ✅ done |
| xCj3 W2: causal patching = negative evidence only | xCj3-W2 | wording commitment (camera-ready) |
| xCj3 W3: scope abstract/title claims | xCj3-W3 | wording commitment (names Qwen2.5 + Qwen1.5) |
| Zici W1: conversational-agent relevance | Zici-W1 | **R01 + R08/R10** ✅ done — OR=75/205 (37%), confirmed at scale |
| Zici W2: layer-selection robustness | Zici-W2 | **R03** ✅ done |
| Zici/JwzU: datasets & software = 1 | Zici comments, JwzU closing | release commitment |

## What's left

**Nothing experimental.** All planned notebooks (R01-R10, skipping the unused R05 outcome)
are run and their numbers are in the docs. Remaining is user-side: read-through, xCj3-W3
title decision, HTML-comment deletion, length-limit check, then posting.

R01–R06 assume the NB14 Colab layout: original memory bank in `./embeddings/`
(from Drive `embeddings/overalign_eval/llama`), NB8 artefacts in `./arditi_artefacts/`.
R01 writes `./rebuttal_embeddings/`, which R02/R04/R05 read. **R07, R08, R10 are independent
of all of this** — each pulls prompts directly from `Sakonii/OveralignDataset` and writes its
own directory (`./qwen_crossmodel_embeddings/`, `./rebuttal_embeddings_v2/`,
`./rebuttal_embeddings_v3/` respectively), never touching R01's `./rebuttal_embeddings/`. R09
reads all LLaMA banks (original + R01 + R08 + R10) plus `arditi_artefacts/` and writes
nothing but figures.

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
- [x] R07 run (2026-07-10): Qwen2.5-7B-Instruct, OR=270/RH=73, cos(OR,RH)=+0.784 CI excludes
      1.0 → JwzU W3 folded in as a genuine result (global §6, JwzU-W3), including honest
      notes on weaker separation and a model-dependent dimensionality flip (86.8% vs <0.9%)
- [x] R08 + R10 run, R09 re-run with the full 4-bank merge (2026-07-10): final full-scale
      numbers folded into JwzU-W1 / global point 1 per R09's decision guide
- [x] Over-refusal definition question (user, 2026-07-10) investigated: generative tasks'
      OR robustness-checked against concerning content types; clean-subset numbers added
      to JwzU-W2 point (4); headline numbers left as-is (consistent with paper's own
      pre-existing methodology)
- [x] Rounding error caught and fixed: continuation's full-scale rate is 48%, not 47%
      (306/644=47.52%, rounds up)
- [x] FINAL polish round (2026-07-10, "final check" user request): standardized artifact
      spelling, removed a contraction and residual meta-phrases ("honestly", "fluke",
      "untapped" x3), fixed the LLaMA comparator inconsistency in W3/global point 6
      (+0.464 from R02 replaced with the CI-backed +0.448 quoted elsewhere in the same
      docs), fixed Zici's sample-count slip (2,600 -> ~505 contents/task), clarified
      "1,705 prompts" as originals + new frames. All greps clean; final word counts:
      global 1,074 / JwzU 1,226 / xCj3 591 / Zici 576.
- [x] Clarity/concision pass (2026-07-10, user request): merged redundant pilot/full-scale
      tables, disambiguated two different cosine quantities in JwzU-W1, fixed stale
      "eight task frames / partial Qwen1.5 replication" wording in xCj3-W3
- [ ] Delete all HTML comments before posting
- [x] No numeric placeholders remain (grep-verified; only the title-suggestion bracket is
      left in xCj3-W3, deliberate)
- [ ] Check ARR response length limits (global response 1,064 words; JwzU 1,232, the
      longest; xCj3 589; Zici 572) — worth a look if there's a hard per-reviewer limit
- [ ] Decide on xCj3-W3 title question (keep or narrow the title — response currently asks for the reviewer's preference)
- [ ] Reconcile the original-bank data-provenance issue before camera-ready (see note above) —
      not reviewer-facing, but needed for internal consistency
- [x] No *technical* general response will be posted (2026-07-11 decision): content folded
      into the three per-reviewer docs; `00_global_response.md` renamed to
      `_NOT_POSTED_global_response_draft.md` and kept only as internal reference
- [x] Overall official comment drafted (2026-07-13, `00_overall_official_comment.md`):
      scores + strengths + thanks only, no technical content duplicated
- [ ] Post: overall official comment first, then three per-reviewer replies (JwzU first)
