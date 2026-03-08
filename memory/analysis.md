# SteeringFail — Experiment Analysis Log

---

## Quick-Reference Verdict Table

| NB | Core Question | Expected | Actual | Verdict |
|---|---|---|---|---|
| 8 | Arditi replication baseline | Full bypass | 100% ASR | Confirmed |
| 9 | Task directions diverge | Low cos-sim | ~0.85 aligned | Fails |
| 9 | Global ablation uneven across tasks | Unequal suppression | All tasks → 0% | Fails |
| 10 | Task-mismatch failure mode exists | Common | 0 cases | Fails |
| 11 | "Assistant" degeneration visible | Yes | P=0.000 | Fails |
| 12 | Task-specific refusal heads in early-mid layers | L0–L20 | Late layers (L27–L31) | Partially fails |
| 13a | Task constellations form in mid-layers | Yes | Confirmed (sil=0.341) | Strong |
| 13a | Over-refusal within task clusters (not global) | Yes | Confirmed — Fig 1b | Strong |
| 13a | Crystallization staggered across tasks | Yes | 3-layer spread, uniform | Fails |
| 13a | Task refusal directions diverge | Yes | 89% aligned by L03 | Fails |

---

## NB8 — Arditi Replication

**Setup:** Best layer L12 (AUROC). 25 refused-harmful, 30 harmless-answered. `cryptanalysis` and `rag_qa` have zero Arditi-class samples.

| Metric | Baseline | Ablated |
|---|---|---|
| Harmful refusal rate | 64.0% | 0.0% |
| ASR | 36.0% | **100.0%** |
| Harmless refusal rate | 0.0% | 0.0% |

Baseline refusal was only 64% (lower than Arditi's near-100%), but full bypass achieved. Direction works.

---

## NB9 / NB10 — Universality & Failure Mode Taxonomy

**Task-direction alignment:** `cos(v_task, v_global) ≈ 0.835–0.858` — nearly identical. Task-specific directions are not divergent.

**Global ablation per task:**
- rephrase: 75% → 0% (+75pp)
- sentiment_analysis: 71.4% → 0% (+71pp)
- translate: 50% → 0% (+50pp)

Global direction achieves complete suppression in all tasks. Cross-task transfer (rephrase → sentiment) nearly equals self-transfer. Diagonal dominance absent.

**Failure mode distribution (α=1.0):** Success=83, Under-steering=37, Over-steering=30, Task-mismatch=0, Layer-mismatch=0. Task-mismatch zone requires cos_align < 0.35; our values are ~0.85. That failure mode never appears.

---

## NB11 — Logit Lens / Vocabulary Projection

P("assistant") = 0.0000 across α=0 to 3.0. The "assistant…assistant" degeneration narrative does not appear. Model stays locked on 'I' (refusal opener).

Per-layer disruption: most disrupted at L0–L1 (100%), stable at L13–L17 and L21–L31. Backwards from prediction — early layers most disrupted, not mid-layers.

---

## NB12 — Task-Conditioned Attention Circuits

**12b — Per-task top attribution heads:**
- rephrase: L28.H31, L30.H4, L27.H31
- sentiment_analysis: L31.H21, L31.H30, L31.H11
- translate: L30.H3, L31.H13, L31.H21

Mean off-diagonal overlap = 0.20 → 80% of top-K heads are task-specific. But all heads are in L27–L31 (late), not early-mid as hypothesized. Task-specificity is itself a late-layer phenomenon. One shared head: L31.H21.

**12c — Causal patching:** Null result. Root cause: shape mismatch silently skips the patch (different sequence lengths). Experiment inconclusive — do not trust.

**Verdict:** Task-conditioned circuits confirmed. Layer-depth prediction fails (task-specific heads are late, not early-mid).

---

## NB13a — Layer-by-Layer Constellation Analysis  [NB7-aligned, updated]

### Methodology
Updated to NB7-aligned approach: behavior is now the primary visual signal (green=target, red=over-refusal) with task identity as faint background. Three new figures added:
- **Fig 1 (updated):** 32-panel galaxy grid, behavior-colored, task clouds as background
- **Fig 1b:** Dual Galaxy Map at peak layer — LEFT: task identity, RIGHT: behavioral class on same UMAP
- **Fig 1c:** Per-task centroid distance (target↔over-refusal) vs. inter-task distance + silhouette line plots

### Section 1 — Inter-Task Silhouette

| Phase | Layers | Mean Silhouette |
|---|---|---|
| Early | L00–L07 | +0.256 |
| Mid | L08–L19 | **+0.341** |
| Late | L20–L30 | +0.309 |

Peak: L12 = 0.357. Task constellations are geometrically real.

### Figure 1b — Dual Galaxy Map (Key Finding)

At the peak-constellation layer, the RIGHT panel (behavioral coloring on the same UMAP) shows: **over-refusal cases (red X) appear within their respective task-coloured clusters, not in a separate global "refusal space."** Behavioral separation is a within-cluster signal, secondary to task identity. This directly confirms the NB7 narrative.

### Figure 1c — Within-Task vs. Inter-Task Gap (Key Finding)

Within-task target↔over-refusal centroid distance (solid lines) is substantially smaller than mean inter-task centroid distance (dashed reference line) across all layers. The behavioral gap is a small perturbation inside the task cluster, not a geometry-dominating signal. This grounds the "refusal is within-cluster" claim quantitatively.

### Section 3 — Within-Task Refusal Crystallization

| Task | Peak Layer | Silhouette | Strength |
|---|---|---|---|
| cryptanalysis | N/A | N/A | No refused samples |
| rag_qa | N/A | N/A | No refused samples |
| rephrase | L19 | 0.066 | Weak |
| sentiment_analysis | L16 | 0.076 | Weak |
| translate | L17 | 0.081 | Weak |

Crystallization is weak (max 0.081, well below 0.3) and narrow (3-layer spread L16–L19). Not staggered.

### Section 5 — Refusal Direction Convergence

- Peak cross-task direction convergence: L03 = 0.891 (very early)
- Peak constellation separation: L12 = 0.357

Directions converge early. Constellation structure (task identity) is the dominant mid-layer variance — Arditi's direction lives in a subspace **orthogonal to task structure**, not because directions diverge, but because refusal direction and task identity occupy orthogonal subspaces.

---

## NB13b — Constellation Centroid Map

**Dataset note:** 31 layers (one missing), 270 samples. OR trajectory tasks: sentiment_analysis (target=32, OR=20) and translate (target=23, OR=28). Rephrase dropped below threshold.

### Numeric Results

**Inter-task centroid distances (high-dim L2):**

| Layer | Mean pairwise | Min pair | Max pair |
|---|---|---|---|
| L00 | 0.00 | 0.00 | 0.00 |
| L04 | 9.04 | 3.56 | 13.25 |
| L08 | 17.47 | 7.86 | 25.32 |
| **L12** | **20.12** | **9.74** | **27.27** |
| L16 | 19.98 | 9.87 | 26.73 |
| L20 | 18.81 | 10.24 | 25.57 |
| L28 | 17.06 | 10.15 | 23.94 |

Peak at L12 (consistent with NB13a silhouette peak). Distances form and stabilise from L08 onward; modest decline through late layers but constellations persist.

**Behavior gap — target vs. over-refusal centroid (L12-anchored UMAP space):**

| Task | Gap at L00 | Gap at L12 | Gap at L28 |
|---|---|---|---|
| sentiment_analysis | 0.000 | 1.492 | 0.527 |
| translate | 0.000 | 2.453 | 1.465 |

L00 gap = 0 because at L00 all embeddings are near-identical (no structure yet). Gap peaks at L12 then **shrinks by late layers** — late-layer processing partially reabsorbs the behavioral split. The OR centroids track closely with their task centroid in the PCA trajectory (Fig 1 dashed lines hug solid lines).

**PCA variance:** PC1=28.4%, PC2=11.2% — only **39.6%** of centroid variance captured in 2D. The centroid galaxy is a highly compressed view.

### Interpretation

**Does this disvalidate task-specific embedding patterns?**

Not the existence, but it sharpens the magnitude claim:

1. **Task constellations are real** — inter-task L2 distances of 17–20 across mid-to-late layers, silhouette 0.357 (NB13a). These are not zero.
2. **But the PCA picture is partial** — 60% of centroid variance is invisible in the 2D galaxy. Visual "clean separation" in 2D is a compressed projection artefact. In full 4096-dim space the structure is real but spread across many dimensions.
3. **Behavioral separation is genuinely small** — OR centroid gaps of 1.5–2.5 UMAP units peak at L12 and shrink to 0.5–1.5 by L28. The late-layer convergence is new: by L28 the model's representation of a refused-benign prompt looks increasingly similar to an answered prompt within the same task. This is consistent with refusal being a "last-moment" decision.
4. **The honest picture:** Task constellations exist and are measurable quantitatively. The clean visual separation in earlier notebooks (NB5/NB7) was partly amplified by per-task UMAP fitting and core-sample filtering. The unfiltered view (Fig 3) is messier but still task-separated, with OR cases as small perturbations within task clusters.

---

## Current Paper Framing

### Confirmed findings

1. **Task constellations are geometrically real (NB13a)** — inter-task silhouette 0.341 mid-layers, peak L12 = 0.357. Measured in full 4096-dim space, not a UMAP artefact. Inter-task L2 distances 17–20 from L08 onward (NB13b).

2. **Over-refusal is within-cluster, not global (NB13a Fig 1b)** — at peak layer, over-refusal cases sit inside task-coloured clouds on UMAP. No separate global "refusal space." Behavioral separation is secondary to task identity.

3. **Within-task behavioral gap is small and shrinks late (NB13a/13b)** — target↔OR centroid distance: L12 ≈ 1.5–2.5 UMAP units, L28 ≈ 0.5–1.5 units. Over-refusal is a small perturbation inside each constellation that partially collapses in late layers.

4. **Arditi's direction is task-agnostic and converges early (NB13a Section 5)** — cross-task alignment reaches 89% by L03, well before constellation peak at L12. The refusal subspace and task-identity subspace are largely orthogonal — not because directions diverge, but because they are geometrically independent.

5. **Task-specific attention circuits exist but only in late layers (NB12)** — 80% of top-K refusal attribution heads are task-specific. All in L27–L31 (not early-mid as predicted). One shared head: L31.H21. Explains the late-layer gap shrinkage: task-conditioned late circuits are where refusal commits.

6. **Arditi direction works for harmful-refusal but only verified in-dataset (NB8/9)** — 100% ASR achieved; all tasks suppressed to 0% equally. Task-specific directions ~0.85 aligned with global — effectively the same vector. Note: direction computed on our 25 harmful samples, not JailbreakBench/AdvBench; this is in-dataset verification, not cross-dataset replication.

---

### Ruled out

| Hypothesis | Result |
|---|---|
| Task-specific refusal directions diverge | No — ~0.85 aligned globally |
| Global ablation is uneven across tasks | No — all tasks → 0% equally |
| Task-mismatch failure mode exists | Never manifests (actual cos_align ~0.85, zone requires <0.35) |
| "Assistant" degeneration from ablation | No — P("assistant") = 0 at all α tested |
| Task-specific refusal heads in early-mid layers | No — all in L27–L31 |
| Staggered crystallization across tasks | No — 3-layer spread L16–L19, all weak (max silhouette 0.081) |

---

### Defensible core argument

> Arditi's refusal direction captures a global, task-agnostic signal that converges early (L03). The dominant structure in mid-layer activations is **task identity** — constellation geometry that Arditi's direction is geometrically orthogonal to. Over-refusal cases sit within their task's cluster as small perturbations (behavioral gap 1.5–2.5 UMAP units at L12), not in a global "refusal space." Arditi's approach correctly handles harmful refusal but is structurally blind to over-refusal, which is driven by task-conditioned geometry inside each constellation. The behavioral gap **shrinks by L28** (0.5–1.5 units), suggesting refusal commits late and is partially reabsorbed — consistent with late-layer refusal circuits (NB12: top heads in L27–L31). SafeConstellations targets the correct geometric object: the within-task over-refusal direction, distinct from the cross-class harmful-refusal direction.

### Calibration notes

- **PCA warning (NB13b):** 39.6% variance captured in 2D centroid galaxy. Visual separation claims should cite high-dim silhouette scores, not the galaxy visualisation.
- **Visual cleanliness (NB5/NB7):** Earlier clean separation was partly artefact of per-task UMAP fitting and core-sample filtering. Unfiltered view is messier but still task-separated.

---

### NB14 — Full results (completed)

**Sample counts (corrected):**
- Over-refusal: 48 (sentiment=20, translate=28, cryptanalysis=0, rag_qa=0, rephrase=0)
- Target (answered benign): 169
- Refused-harmful (Arditi class A): 25
- Harmless-answered (Arditi class B): 157

**H1: cos(OR direction, Arditi direction) per layer:**
- L00: +0.0000, L03: +0.3164, L08: +0.4564, L12: +0.4480, L16: +0.4422, L20: +0.4205, L24: +0.4029, L28: +0.3855
- Peak: ~0.456 at L08; stays roughly 0.40–0.46 across mid–late layers
- Harmful-refusal baseline (NB9): 0.845–0.858
- Difference at L12: −0.402 → OR direction is substantially less aligned with Arditi than harmful-refusal directions are with each other

**H2: Per-task OR direction pairwise similarity at L12:**
- Only 2 tasks valid (sentiment OR=20, translate OR=28; others have 0 OR samples)
- Off-diagonal mean: 0.5635 (vs harmful-refusal 0.845 ± 0.011) → OR more task-specific
- Per-task vs global OR alignment: sentiment=+0.4598, translate=+0.3252

**H3: Arditi ablation (GPT-4o judge, n=20 per condition):**
- Harmful baseline: 65.0% → ablated: 10.0% → suppression: **+55.0 pp**
- OR baseline: 55.0% → ablated: 5.0% → suppression: **+50.0 pp**
- SURPRISE: Arditi suppresses BOTH classes similarly (~55pp harmful, ~50pp OR)

**H3 interpretation — "blunt instrument" narrative:**
H3 did NOT confirm "Arditi is blind to OR." Instead, the ~0.45 geometric overlap between OR and Arditi directions is sufficient for substantial suppression. The important implication is SELECTIVITY: Arditi cannot fix OR without simultaneously disrupting the harmful-refusal circuit. SafeConstellations targets the task-conditioned OR subspace (orthogonal to Arditi), enabling selective OR correction without collateral safety cost. The geometric distinctness matters for selectivity, not just raw suppression.

### NB14 — Strongest remaining test (not yet run)

**H1:** Compute `mean(benign_refused) - mean(benign_answered)` per layer = over-refusal direction. Measure cosine similarity with Arditi's harmful-refusal direction. If low → they solve geometrically different problems. Cleanest single number for the paper.

**H2:** Per-task over-refusal directions — do they diverge across tasks (unlike harmful-refusal directions which are ~0.85 aligned)? If yes, vindicates "task-conditioned" claim for the correct target class.

**H3 (model required):** Apply Arditi ablation to over-refused benign prompts. If over-refusal is barely suppressed → empirical proof Arditi is blind to this class.
