# Qwen1.5-7B Cross-Model Analysis

Mirrors the LLaMA-3.1-8B analysis notebooks for Qwen1.5-7B.
All notebooks work from pre-computed embeddings — **no GPU required**.

## Setup

Set `EMBEDDINGS_DIR` in each notebook's config cell to point to your Qwen embeddings directory.
The directory must contain one `.csv` file and its associated `.pt` tensor file
(same format as produced by `4. [Embedding-Qwen1.5] Memory Bank Construction.ipynb`).

**Colab:** Uncomment the Colab cell in each notebook and set your Drive path.

**Local:** Download `embeddings_qwen/` from Drive and set the path directly.

## Notebooks

| Notebook | LLaMA mirror | Status |
|---|---|---|
| `Q13a_Constellation_Analysis.ipynb` | NB13a | Full — silhouette, galaxy PCA, CDist |
| `Q14_OR_Direction_Analysis.ipynb` | NB14 | Partial — OR direction only (no RH: n=1) |
| `Q16_OR_Subspace_Dimensionality.ipynb` | NB16 | Partial — OR subspace only |
| `Q17_Linear_Probing.ipynb` | NB17 | Partial — task + behavior probes (type probe skipped) |

## Key Qwen Differences vs LLaMA

| Property | LLaMA-3.1-8B | Qwen1.5-7B |
|---|---|---|
| Layers | 32 | 31 |
| Constellation peak | L12 (silhouette=0.454) | L5 (silhouette=0.458) |
| OR samples | 48 | 22 |
| RH samples | 25 | 1 ← major limitation |
| OR tasks | sentiment + translate | cryptanalysis (n≥5 only) |

## What Can and Cannot Be Replicated

**Claim 1 — Task-identity constellations:** ✅ Fully replicable (Q13a)

**Claim 2 — OR direction misalignment (cos ~0.45):** ❌ Cannot compute
— Qwen RH n=1, so the harmful-refusal DIM direction is undefined.

**Claim 3 — OR subspace is higher-dimensional:** ⚠️ Partial
— OR subspace computed; comparison to RH subspace not possible.

## Output Figures

All figures saved to `./figures/` with prefix `q_`.
