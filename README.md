# MgAir-Multimodal

**Multimodal surrogate modeling and high-throughput alloy screening for Mg-air battery anodes.**

This repository contains the core workflow used to build the BatteryBERT-GB fusion model, identify current-density-specific element pools, screen quaternary Mg alloys, interpret model predictions.


## Research Workflow

The computational workflow consists of four main stages:

1. **Multimodal prediction**  
   Combine structured composition-processing descriptors with BatteryBERT text embeddings through late-fusion stacking.
2. **Element-pool identification**  
   Rank experimental alloys at 2.5, 10, and 40 mA cm⁻² using normalized voltage and anodic-efficiency objectives.
3. **High-throughput screening**  
   Search current-specific quaternary composition-processing spaces and rank candidates using a composite z-score.
4. **Interpretation and visualization**  
   Analyze SHAP contributions and generate publication figures and source-data workbooks.

## Repository Layout

```text
MgAir-Multimodal/
|-- scripts/           # Modeling, screening, interpretation, and plotting code
|-- .gitignore         # Model artifacts, caches, and generated figures
|-- requirements.txt   # Python dependencies
`-- README.md          # Project documentation
```

All commands below should be run from the repository root.

## Core Pipeline

These scripts form the primary computational workflow and should be executed in the listed order.

| Priority | Script | Role |
|---|---|---|
| 1 | `scripts/bertmutimodelstack.py` | Trains and evaluates the BatteryBERT-GB late-fusion stacking model. Produces five-fold OOF predictions, metrics, and reusable model outputs. |
| 2 | `scripts/Find_element313.py` | Performs dual-objective ranking at 2.5, 10, and 40 mA cm⁻² and produces for element-pool construction. |
| 3 | `scripts/filter_251040_Ternary.py` | Runs the final quaternary Mg-alloy search across current-specific composition and processing grids. |

### Filename Note

Although `filter_251040_Ternary.py` contains `Ternary` in its historical filename, it uses:

```python
combinations(element_pool, 3)
```

This represents an Mg matrix plus three alloying additions and therefore corresponds to a **quaternary alloy system**.

## Publication Figures

| Script | Output |
|---|---|
| `scripts/build_figure2_dataset_distribution.py` | Figure 2: dataset distributions, element occurrence, process coverage, and electrochemical-property relationships. |
| `scripts/build_figure3_reference_style.py` | Figure 3: comparison of four numeric, two text-only, and two multimodal models; OOF parity plots and sample-level errors. |
| `scripts/build_figure4_explainability.py` | Figure 4: global and current-density-specific SHAP interpretation of the final BatteryBERT-Fusion model. |
| `scripts/build_figure5_quaternary_screening.py` | Figure 5: quaternary-system rankings, candidate distributions, and representative pathways at 2.5, 10, and 40 mA cm⁻². |
| `scripts/split_figure5_subfigures.py` | Replots Figure 5 panels (a)-(i) as separate PNG, PDF, and SVG files. |

## Supporting and Reference Code

| Script | Purpose |
|---|---|
| `scripts/build_paper_package.py` | Provides OOF-result loading and reconstruction utilities used by `build_figure3_reference_style.py`. |
| `scripts/filter_251040_3emement.py` | Retained ternary-alloy control workflow using an Mg matrix plus two alloying additions; included for methodological comparison and traceability. |

## Required Inputs

Place the following local inputs in the **repository root** before running the corresponding stages:

| Input | Purpose |
|---|---|
| `rawdata.xlsx` | Experimental composition, processing, current-density, voltage, and anodic-efficiency data. |
| `outputs_GBstack_hpo/` | Previously trained stacking-model results and OOF outputs, required when regenerating figures without retraining. |
| `outputs_design_search_multiCD_selected/` | Trained screening models and retained high-throughput candidates. |
| Local BERT and BatteryBERT cache | Transformer tokenizer and encoder weights used by the text and fusion branches. |

These inputs are intentionally excluded from version control. Generated models, figures, caches, and intermediate outputs are also excluded through `.gitignore`.

## Environment Setup

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

GPU acceleration is optional but recommended for Transformer embedding generation and large screening runs.

## Execution

### Full Training and Quaternary Screening

```powershell
python scripts/bertmutimodelstack.py
python scripts/Find_element313.py
python scripts/filter_251040_Ternary.py
```

### Reproduce Figures 2-5

Run these commands after the required model and screening outputs are available:

```powershell
python scripts/build_figure2_dataset_distribution.py
python scripts/build_figure3_reference_style.py
python scripts/build_figure4_explainability.py
python scripts/build_figure5_quaternary_screening.py
python scripts/split_figure5_subfigures.py
```

## Outputs

| Directory | Contents |
|---|---|
| `paper_figures/` | Final Figures 2-5 and figure source-data workbooks. |
| `paper_figures/Figure5_subfigures/` | Standalone Figure 5 panels in PNG, PDF, and SVG formats. |
| `outputs_figure3_8models/` | Eight-model OOF predictions, metrics, and intermediate Figure 3 results. |
| `outputs_figure4_explainability/` | SHAP values, caches, and Figure 4 intermediate results. |
| `outputs_GBstack_hpo/` | Stacking-model artifacts and OOF evaluation outputs. |
| `outputs_design_search_multiCD_selected/` | High-throughput screening candidates and reusable model artifacts. |

## Reproducibility Notes

- Use the same version of `rawdata.xlsx` for model training, SHAP interpretation, and figure generation.
- Figure 3 depends on helper functions in `scripts/build_paper_package.py`; both scripts must remain in `scripts/`.
- Figure 3 and Figure 4 can load Transformer models from the local Hugging Face cache for offline execution.
- Full high-throughput screening is computationally intensive; reuse existing model artifacts when only regenerating figures.
