# SteeringFail Project Memory

## Project Overview
**Successor Paper:** "Why Steering Methods Fail? A Mechanistic Analysis from Safety Perspective"
- Central thesis: LRH is incomplete — refusal is task-conditioned, but Arditi's direction is geometrically orthogonal to task structure, not because directions diverge per task but because the refusal subspace and constellation subspace are largely orthogonal.
- SafeConstellations targets within-task over-refusal geometry; Arditi targets cross-class harmful-refusal geometry. Different problems, different directions.

## Dataset / Model
- Model: meta-llama/Meta-Llama-3.1-8B-Instruct (32 layers, 4096-dim)
- Dataset: 270 samples × 32 layers
- Tasks: sentiment_analysis, translate, cryptanalysis, rag_qa, rephrase
- Labels: response_labels (cautious/harmful/not_harmful) + refusal_class (direct_answer/direct_refusal/indirect_refusal)
- TARGET_MASK: (cautious | not_harmful) & direct_answer
- OVER_REFUSAL_MASK: refusing & benign_intent_tasks (sentiment, translate, cryptanalysis, rag_qa)
- Embeddings: ./embeddings/ — CSV (torch_path, llm_evaluation, refusal_class) + .pt files (layer_{n}_input_norm, final_norm)

## Per-Notebook Summaries

**NB1–6 (Existing work):** Dataset construction, LLM-based evaluation pipeline, SafeConstellations steering method. Established task taxonomy (5 tasks), response labels, refusal_class annotations, embedding extraction (32 layers × 4096-dim, stored as .pt files).

**NB7 — Clustering Analysis (reference methodology):**
Per-layer UMAP fitted on all 270 samples together; 32 panels, each showing all tasks simultaneously. Galaxy Map (dual view): LEFT=task identity, RIGHT=behavior on same UMAP at peak-separation layer. Per-task silhouette and centroid metrics in high-dim space. This is the gold standard for NB13a.

**NB8 — Arditi Replication:**
Direction extracted at L12 (best AUROC). 25 refused-harmful, 30 harmless-answered. Baseline refusal 64% (not near-100%); ablation → 100% ASR. Global direction works for harmful-refusal. Cryptanalysis and rag_qa have 0 Arditi-class samples.

**NB9 — Universality Test:**
Task-specific Arditi directions vs. global: cos_align = 0.835–0.858 (nearly parallel). Global ablation achieves 0% refusal in all tasks (rephrase 75→0%, sentiment 71→0%, translate 50→0%). Cross-task transfer nearly equals self-transfer — no diagonal dominance.

**NB10 — Failure Mode Taxonomy:**
At α=1.0: Success=83, Under-steering=37, Over-steering=30, Task-mismatch=0, Layer-mismatch=0. Task-mismatch zone requires cos_align < 0.35; actual values ~0.85 → failure mode never manifests. Taxonomy collapses to under/over-steering only.

**NB11 — Logit Lens / Vocabulary Projection:**
P("assistant") = 0.0000 across α=0–3.0; model stays locked on 'I'. No "assistant…assistant" degeneration. Most disrupted layers: L0–L1 (100%). Stable: L13–L17 and L21–L31. Backwards from prediction — early layers, not mid, are most sensitive.

**NB12 — Task-Conditioned Attention Circuits:**
12a: Refusal attribution localises to specific heads, peak L31.H21=0.031. 12b: mean off-diagonal overlap=0.20 → 80% of top-K heads task-specific, but ALL in L27–L31 (not early-mid as predicted). One shared head: L31.H21. 12c: Causal patching null result — shape mismatch bug silently skips patches; inconclusive.

**NB13a — Layer-by-Layer Constellation Analysis [NB7-aligned]:**
32-panel behavior-colored galaxy grid (green=target, red=over_refusal). Inter-task silhouette: early=0.256, mid=0.341, late=0.309; peak L12=0.357. Fig 1b Dual Galaxy Map: over-refusal (red X) sits INSIDE task-colored clouds — not in global refusal space. Fig 1c: within-task target↔OR centroid gap << inter-task distance at all layers. Within-task behavioral silhouette weak (max 0.081 at L16–L19). Refusal directions converge 89% cross-task by L03.

**NB13b — Constellation Centroid Map:**
PCA of task+OR centroid trajectories across all 31 layers (PC1=28.4%, PC2=11.2%; only 39.6% variance). Inter-task L2 distances peak at L12 (mean=20.12), stable L08+. Behavior gap (target↔OR centroid, UMAP units): sentiment L12=1.492→L28=0.527; translate L12=2.453→L28=1.465. Gap SHRINKS late → refusal partially reabsorbed by L28, consistent with NB12 late-layer heads. Visual galaxy separation is partly a compressed projection artefact; silhouette scores in high-dim are the reliable metric.

## Key Confirmed Findings
1. Task constellations are real — silhouette 0.341 at mid-layers, peak L12 = 0.357
2. Over-refusal cases sit WITHIN task clusters (Fig 1b NB13a) — behavioral separation is secondary to task identity
3. Within-task target↔OR centroid distance << inter-task distance (Fig 1c NB13a)
4. Arditi direction converges cross-task by L03 (89%) — orthogonal to constellation structure, not divergent
5. Refusal crystallization weak (max silhouette 0.081) and narrow (L16–L19 only)
6. Task-specific attention heads exist (80% unique) but in L27–L31, not early-mid

## Key Failed Hypotheses
- Task-specific refusal directions diverge: No — ~0.85 aligned globally
- Staggered crystallization: No — 3-layer spread, all L16–L19
- "Assistant" degeneration: No — P=0.000 at all tested α
- Task-mismatch failure mode: Never manifests

## NB14 Results (completed)
- Sample counts: OR=48 (sentiment=20, translate=28), TARGET=169, REFUSED_HARMFUL=25
- H1: cos(OR dir, Arditi dir) = 0.448 at L12 (vs harmful-refusal baseline 0.845–0.858). Substantially lower but NOT near-zero (~45° vs ~30° angle)
- H2: Per-task OR direction alignment = 0.564 (vs 0.845 for harmful-refusal) — OR more task-specific; only 2 tasks valid
- H3: Arditi suppresses harmful 65%→10% (+55pp) AND OR 55%→5% (+50pp) — similar magnitude on both
- KEY REVISION: H3 does NOT support "Arditi blind to OR." Instead: Arditi is a BLUNT instrument that suppresses both. The geometric distinctness matters for SELECTIVITY (SafeConstellations can fix OR without touching harmful-refusal circuit), not raw suppression.
- Paper narrative updated: "non-selective suppression" is the problem, not zero suppression

## NB15 Results (completed)
- SafeConstellations vs Arditi head-to-head (n=20 per condition, GPT-4o judge, seed=42)
- OR: baseline 55% → Arditi 5% (−50pp) | SafeConstel 0% (−55pp)
- Harm: baseline 65% → Arditi 10% (−55pp) | SafeConstel 30% (−35pp)
- Selectivity score (OR-supp/harm-supp): Arditi 0.91, SafeConstellations 1.57
- Mechanistic explanation for 35pp harm bypass: harmful samples are task-wrapped (translate/rephrase/sentiment), share constellation at L12 with OR samples → task hook fires on task identity, not content. Explained by NB13a finding.
- Steering layers: [10,11,12,13,14], alpha=1.0, additive hook

## Paper (paper/acl_latex.tex)
- Title: "Over-Refusal and Subspaces: A Geometrical Analysis of Task-Conditioned Refusal in Aligned LLMs"
- Venue target: EMNLP (ACL format)
- Storyline: Harmful-refusal (global, early, 1D, Arditi) vs. over-refusal (within-task, mid-layer, task-conditioned). Arditi is blunt (0.91 selectivity). SafeConstellations is more selective (1.57).
- 6 figures: fig1_galaxy_map, fig3_silhouette_gap, fig2_cosine_h1, fig4_h2_or_tasks, fig5_h3_ablation, fig6_selectivity (NB15 → fig_nb15_selectivity.png)
- New §4.6: SafeConstellations vs Arditi (NB15) with Table 3 and Fig 6
- §6 Planned Experiments remaining: multi-dim probing, NB12c fix, cross-model replication
- No placeholders remaining in main body; limitations section updated

## Files
- analysis.md: Full results, verdicts, and current paper framing (authoritative)
- analysis.txt: Older raw notes (superseded by analysis.md)
- research_plan.md: Original notebook designs
