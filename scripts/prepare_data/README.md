# Prepare Data for ArboEL

This CLI prepares the data required for training and evaluating ArboEL. It builds dictionary pickles from UMLS concepts and converts entity linking datasets into the JSONL format expected by the encoder.

## Pipeline Overview
1.  **Prepare Dictionaries**: Reads processed UMLS parquet files (`all_disambiguated.parquet`) to create a dictionary of all possible entities (CUIs) and their descriptions. Saves this as `dictionary.pickle` for each dataset.
2.  **Convert Datasets**: Converts BigBio datasets (MedMentions, QUAERO, SPACCC) and synthetic datasets into a JSONL format suitable for the bi-encoder. Each line represents a mention with its surrounding context and label.

---

## Usage

### 1. Standard Run
Run the full pipeline for all default datasets (MedMentions, EMEA, MEDLINE, SPACCC).

```bash
python scripts/3b_prepare_data_encoder/run.py
```

### 2. Customizing Paths
If your data is located in non-standard directories (e.g., if you are using specific synthetic data files), you can override the paths.

```bash
python scripts/3b_prepare_data_encoder/run.py \
  --datasets MedMentions EMEA \
  --out-root arboEL/data/final_data_encoder \
  --umls-mm-path data/UMLS_processed/MM \
  --synth-mm-json data/synthetic_data/SynthMM/SynthMM_bigbio.json
```

---

## Key Arguments

| Option | Default | Description |
| :--- | :--- | :--- |
| `--datasets` | `['MedMentions', 'EMEA', 'MEDLINE', 'SPACCC']` | List of datasets to process. |
| `--out-root` | `arboEL/data/final_data_encoder` | Root directory for the output JSONL and pickle files. |
| `--umls-mm-path` | `data/UMLS_processed/MM` | Path to the directory containing MedMentions UMLS data (`all_disambiguated.parquet`). |
| `--umls-quaero-path` | `data/UMLS_processed/QUAERO` | Path to the directory containing QUAERO UMLS data. |
| `--umls-spaccc-path` | `data/UMLS_processed/SPACCC` | Path to the directory containing SPACCC UMLS data. |

---

## Outputs

For each dataset, the script generates a folder in `--out-root` containing:

*   **`dictionary.pickle`**: A pickled dictionary mapping CUIs to their title and description.
*   **`train.jsonl`**, **`valid.jsonl`**, **`test.jsonl`**: The processed dataset splits in JSONL format.

**Example Directory Structure:**
```text
arboEL/data/final_data_encoder/
├── MedMentions/
│   ├── dictionary.pickle
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── EMEA/
│   ├── dictionary.pickle
│   └── ...
└── ...
```

### JSONL Format
Each line in the output `.jsonl` files contains a JSON object with the following fields:

*   `mention`: The text of the entity mention.
*   `mention_id`: A unique ID for the mention (DocumentID.EntityID).
*   `context_left`: Text immediately preceding the mention.
*   `context_right`: Text immediately following the mention.
*   `context_doc_id`: The ID of the document.
*   `type`: The semantic group of the entity (e.g., "Chemicals & Drugs").
*   `label_id`: The correct CUI (gold standard label).
*   `label`: The description of the correct CUI from the dictionary.
*   `label_title`: The title of the correct CUI.
