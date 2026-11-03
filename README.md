# TELLER: Trustworthy Misinformation Detection

This repository contains the research code for **TELLER: A Trustworthy Framework for Explainable, Generalizable and Controllable Fake News Detection**.

## Quick start: run a local demo

The small Flan-T5 model downloads automatically from Hugging Face the first time you run it. This demo does not need the research dataset or an OpenAI key.

1. Install Python 3.11 or newer.
2. Create and activate a virtual environment from the repository folder:

   **Windows PowerShell**

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

   **macOS / Linux**

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

3. Install PyTorch for your computer. For a CPU-only setup:

   ```bash
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements.txt
   ```

   For an NVIDIA GPU, choose the matching CUDA command from the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install `requirements.txt`.

4. Run a claim through the local model:

   ```bash
   python predict.py "The claim you want to check goes here."
   ```

   To include context:

   ```bash
   python predict.py "The claim you want to check goes here." --evidence "Relevant source or context."
   ```

The first model download is several hundred megabytes. The prediction is a research demo output, not a verified fact-check.

## Reproduce the research pipeline

The training scripts use the preprocessed datasets and generated question/answer files from the original project. Download [data.zip](https://portland-my.sharepoint.com/:u:/g/personal/liuhui3-c_my_cityu_edu_hk/EfApQlFP3PhFjUW4527STo0BALMdP16zs-HPMNgwQVFWsA?e=zoHlW2), extract it into the repository's `data/` directory, and keep the dataset subfolders (for example `data/LIAR-PLUS/` and `data/Constraint/`). Each dataset folder needs the JSONL splits and the question/answer JSON files referenced by the scripts.

Run the in-domain LIAR pipeline:

```bash
python drive_liar.py --dataset_name LIAR-PLUS --data_path data --device cpu
```

Run cross-domain experiments:

```bash
python drive_dg.py --data_path data --device cpu
```

Use `--device auto` to use CUDA when PyTorch detects a supported GPU, otherwise CPU. The research runs can take substantially longer than the quick demo and require all of the preprocessed files described above.

## Optional OpenAI API mode

Local Flan-T5 use does not require an API key. For code paths that use an OpenAI model, set `OPENAI_API_KEY` in your shell or create a local `.env` file based on `.env.example`. Never commit `.env` or a real key. Set `TELLER_OPENAI_MODEL` to a model enabled for your API account; the default is `gpt-5.5`.

## Model storage

Hugging Face caches models in its normal user cache. To use a pre-downloaded Flan-T5 model, put it under `llms/flanT5/<model-name>/` or set `TELLER_FLAN_T5_DIR`. Llama 2 is optional and requires accepting Meta's model license and Hugging Face access before downloading; set `TELLER_LLAMA_DIR` to use a local copy.
