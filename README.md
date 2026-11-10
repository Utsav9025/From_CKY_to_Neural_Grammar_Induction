# From CKY to Neural Grammar Induction

NLP Assignment 2, Project 3. The three tasks use the same Penn Treebank Partition A split.

## Environment

Use Python 3.12. Create a virtual environment and install the pinned packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead of `source`. The separate PyTorch command installs the CPU build used for the reported evaluation. For GPU training, use a CUDA build of PyTorch 2.9.1 and skip the CPU install command. Run the notebooks in Jupyter, VS Code, or Colab with the environment selected. The notebooks select CUDA when available.

## Data preparation

All three tasks use the NLTK distribution of the Penn Treebank sample. Download it with `python -m nltk.downloader treebank`, or let the notebook's data preparation cell download it. No separate corpus files are needed.

1. Shuffle the 3,914 tree indices with NumPy seed 42. Partition A is the first 1,305 indices; B and C are the next 1,305 and 1,304. This submission uses A.
2. Shuffle A again with seed 42. The split has 1,044 training trees, 130 validation trees, and 131 test trees. Each run records the exact index lists in `partition_indices.json`.
3. Remove `-NONE-` traces and strip functional suffixes from category labels. Task 1 uses the cleaned trees for counting; Task 2 uses their word sequences and skips one single-word training sentence; Task 3 uses the cleaned gold trees.

The global Python, NumPy, and PyTorch seed is 42. Task 2 uses `42 + epoch` for batch order and 1234 for repeatable validation samples. Task 3 uses 42 for its first training order and `42 + epoch` for later orders.

## Run the tasks

Open the notebook for the desired task and run it from the first cell. Each task notebook includes its own preparation and evaluation.

| Task | Notebook | What it runs |
| --- | --- | --- |
| 1 | `code/task1/Task_1_Counted_PCFG.ipynb` | Fits counted grammars for Markov orders 0 and 1, selects the rare-word cutoff on validation, and evaluates CKY on test. |
| 2 | `code/task2/Task_2_Compound_PCFG.ipynb` | Loads or trains the Compound PCFG, selects Viterbi or MBR on validation, and evaluates on test. |
| 3 | `code/task3/Task_3_Supervised_Neural_PCFG.ipynb` | Loads or trains the supervised neural grammar and scores its CKY parses on validation and test. |

To run the whole project in sequence, use `code/Final_Grammar_Induction_Notebook.ipynb`. It runs Task 1, Task 2, Task 3, the optional counted/neural experiment, and the final comparison under the same scorer.

### Saved models and fresh training

The report uses these two checkpoints trained for this project. No pretrained external weights were used.

| Model | Checkpoint | Source |
| --- | --- | --- |
| Task 2 | `task2_best_epoch1_saved.pt` | [Download from Drive](https://drive.google.com/file/d/1yDXrNZwlQzL2ifp_vFX9xEn0WlC8G4vL/view); lowest validation loss in the batch-4 run. |
| Task 3 | `task3_epoch15.pt` | [Download from Drive](https://drive.google.com/file/d/1TSyhTKk_6pZg8I6Lz2jJtEObzL6RwpCU/view); epoch 15 of supervised training. |

The default settings load the files from `model_dir` (`checkpoints/` in the kernel's working directory), or download them from the [project folder](https://drive.google.com/drive/folders/1dfAnwiUpgn5CSfKkdGzWyLrVaNogQfoS?usp=drive_link). For manual downloads, point `model_dir` to the folder containing the files and set `download_models = False`.

For training from scratch, set `train_compound_model = True` for Task 2 and/or `train_supervised_model = True` for Task 3 in **Settings**, then run from the first cell. Both train for 15 epochs. Task 2 evaluates its best validation checkpoint and Task 3 evaluates epoch 15. New runs can produce different scores from the report's saved models.

## Outputs

Each notebook writes split indices, a run manifest, metrics, and predicted trees to `results/<run_timestamp>/` under the kernel's working directory. Freshly trained weights go into that run's `trained_checkpoints/` subfolder. The combined notebook also writes `final_comparison.csv`, `hybrid_summary.json`, the plots, and a results ZIP.

`report.pdf` explains the methods, the shared evaluation, the paper comparison, and examples of correct and incorrect parses. Its main comparison uses the original Task 3 neural grammar; the counted/neural mixture is reported separately.
