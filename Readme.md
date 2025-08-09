Project: Understanding the importance of punctuations in information propogation. Token level interventions and layer swapping experiments to evaluate the model reasoning ability. Experiments done on gpt2, gemma-2b, and deepseek-1.3b.

- **Trained Models**:
  - `models/`: Includes zipped model files.  
    - `gpt2`: finetuned gpt2 model can be unzipped and used directly.  
    - `deepseek`, `gemma`: LoRA adapters. Merge with base models using `scripts/run_merge.py`.  

- **Datasets**:  
  - Available in `Datasets_intervention/`

- **Visualizations**:  
  - All experiments generate visualizations automatically.

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/anonymous/reasoning-interventions.git
cd reasoning-interventions
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

Make sure you also install:
- `nnsight`
- `transformers`
- `peft`
- `huggingface_hub`

---
## Merging Adapter with Base Model (Gemma / DeepSeek)

To replicate results with Gemma or DeepSeek models, use the `run_merge.py` script:

```bash
python run_merge.py \
    --zip_path "models/gemma_adapter.zip" \
    --base_model "google/gemma-2b" \
    --output_path "models/gemma_merged" \
    --hf_token "your_huggingface_token"
```

This script:
- Unzips the LoRA adapter.
- Loads the base model from HuggingFace.
- Applies the adapter and saves the full merged model.

---
## Running Experiments

### 1. Token-Level Intervention

To run interchange intervention experiments:

```bash
cd scripts
python run_intervention.py \
    --model_path <model_path> \
    --dataset_dir <directory having dataset> \ 
    --intervention_type <type of intervention being done> \ 
    --model_name <GPT2/DeepSeek/Gemma>
```

---

### 2. Rule-Based Reasoning Experiments

To run rule intervention experiments:

```bash
cd scripts
python rule_intervention.py \
    --model_name <gpt2,gemma,deepseek> \
    --model_path <model_path> \ 
    --dataset_dir "datasets_intervention/<dataset_name>.json" \
    --rule_type <all_rule,if_then_rule>

```

---

### 3. Necessity & Sufficiency Ablation

Scripts available in `ablation/`:

- **run_gpt2_single.py** – for last-layer ablations
- **run_incremental_ablation.py** – for incremental layer removals
- **run_necessity_sufficiency.py** – test necessity and sufficiency of punctuation tokens

Example:

```bash
python run_gpt2_singl_layer_ablation.py \ 
    --model_path <model_path> \
    --output_path <results_storage_path> \
    --model_name <gpt2,deepseek,gemma>
```

---

### 4. Rule-Based Layer Swap

```bash
python run_layer_swap.py \
    --swap_type <if_then,all> \
    --model_path <model_path> \
    --model_name <model_name> \
    --dataset_dir <dataset_dir>

```
---

## Visualization

All experiments generate CSVs and plots in `visualisations/`. 

## Datasets

All datasets are present in:

```
datasets_intervention/
```

- `rule_taker_subject.csv` – for subject intervention
- `rule_taker_adjective.csv` – for adjective intervention
- `rule_taker_full_sentence.csv` – for dot, full sentence, necessity and sufficiency experiments
- `two_sent_check.csv` - for two sentence swaps.
- `rule_inference.csv` - `if_then` rule analysis
- `rule_inference.csv` - `All` rule analysis
- `layer_swap_all.csv` - layer swap analysis for `All` rules.
- `layer_swap_if_then.csv` - layer swap analysis for `if_then` rules.
---
