Mg-air battery paper: core code
Multimodal fusion surrogate model

Core workflow
bertmutimodelstack.py

Trains the BatteryBERT-GB late-fusion stacking model.

Generates 5-fold OOF (out-of-fold) predictions, evaluation metrics, and outputs_GBstack_hpo results.

Find_element313.py

Performs dual-objective normalized ranking on experimental data at 2.5, 10, and 40 mA cm^-2.

Generates top5_alloys_summary.xlsx to determine the candidate element pools for each operating condition.

filter_251040_Ternary.py

The quaternary alloy search script used in the final paper.

Note: Although the historical filename contains Ternary, the code uses combinations(element_pool, 3)—representing the Mg matrix plus 3 alloying elements, which corresponds to quaternary alloys.

filter_251040_3emement.py

The retained ternary alloy control script, using the Mg matrix plus 2 alloying elements.

Since the final screening in the current paper primarily focuses on quaternary systems, this script is mainly used for methodological traceability and comparison.

build_figure2_dataset_distribution.py

Plots dataset distribution, element frequency, process coverage, and electrochemical performance relationships (Figure 2).

build_figure3_reference_style.py

Compares 4 purely numerical models, 2 purely text-based models, and 2 multimodal fusion models.

Plots OOF metrics, parity plots, and sample-level errors (Figure 3).

build_figure4_explainability.py

Performs hierarchical SHAP and current-density-wise SHAP analysis on the best-performing BatteryBERT-Fusion model (Figure 4).

build_figure5_quaternary_screening.py

Organizes quaternary high-throughput screening results, generates alloy system rankings, as well as candidate scatter plots and representative system path plots at 2.5, 10, and 40 mA cm^-2 (Figure 5).

split_figure5_subfigures.py

Reads the Figure 5 source data workbook and replots panels (a)–(i) into 9 standalone PNG, PDF, and SVG subfigures.

build_paper_package.py

Helper module for reading and reconstructing OOF results used in Figure 3.

Currently, build_figure3_reference_style.py imports its get_oof_bundle function, so it must be placed in the same directory as the Figure 3 script.

Required inputs
The scripts retain the original project relative paths. To run them independently within this directory, place the following input files and directories at the same level as the scripts:

rawdata.xlsx

outputs_GBstack_hpo/

outputs_design_search_multiCD_selected/

BERT and BatteryBERT weights in the local Hugging Face cache

This package only includes core code without duplicating the aforementioned large datasets and model artifacts. Original inputs remain in the parent project directory.

Recommended run order
Full retraining and screening:

PowerShell
python bertmutimodelstack.py
python Find_element313.py
python filter_251040_Ternary.py
Replotting paper figures based on existing models and screening results:

PowerShell
python build_figure2_dataset_distribution.py
python build_figure3_reference_style.py
python build_figure4_explainability.py
python build_figure5_quaternary_screening.py
python split_figure5_subfigures.py

Outputs
Final Figures 2–5: paper_figures/

Figure 3 intermediate results: outputs_figure3_8models/

Figure 4 SHAP results: outputs_figure4_explainability/

Figure 5 standalone subfigures: paper_figures/Figure5_subfigures/

Stacking model and OOF results: outputs_GBstack_hpo/

High-throughput screening and model artifacts: outputs_design_search_multiCD_selected/