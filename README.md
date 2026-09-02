---
title: DocuChat AI
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
license: mit
---

# DocuChat AI — Fine-Tuned PDF Question-Answering Assistant

DocuChat AI is a simple final-year GenAI project. Users upload a text-based PDF,
the system cleans and divides its text, generates 384-dimensional MiniLM
embeddings, creates an in-memory FAISS index, retrieves the top three passages,
and produces a grounded answer with page citations.

The repository also includes QLoRA fine-tuning for a compact pretrained LLM,
prompt engineering, Streamlit deployment, Docker, tests, and GitHub Actions.

## Architecture

```text
PDF upload → PyMuPDF extraction → cleaning → 600-character chunks
→ MiniLM embeddings → FAISS cosine search → top-3 passages
→ grounded prompt → Qwen2.5-0.5B or LoRA adapter → answer + [Page N]
```

## Models

- Generator: `Qwen/Qwen2.5-0.5B-Instruct`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Fine-tuning: QLoRA with 4-bit NF4, PEFT, and TRL

The default backend is `extractive`, so the deployed demo is fast and does not
need to load the LLM. Set `GENERATOR_BACKEND=transformers` to use the base model
or the adapter at `models/docuchat-lora`.

## Run locally

Use Python 3.11:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-app.txt
streamlit run app.py
```

The first run downloads the embedding model. Open `http://localhost:8501` when
running Streamlit locally.

## Use the LLM backend

After copying a trained adapter into `models/docuchat-lora`:

```bash
# PowerShell
$env:GENERATOR_BACKEND="transformers"
streamlit run app.py

# Linux/macOS
GENERATOR_BACKEND=transformers streamlit run app.py
```

If the adapter directory is absent, Transformer mode loads the base model. This
makes it easy to compare base and fine-tuned responses.

## Fine-tune

Follow `COLAB_GUIDE.md`. The trainer uses:

- 4-bit NF4 quantization and double quantization;
- LoRA rank 8 and alpha 16;
- two epochs by default;
- 512-token sequences;
- gradient checkpointing and accumulation.

`data/training_data.jsonl` contains five schema examples only. Create and manually
review 100–300 examples before training. Do not claim a dataset size or accuracy
that you have not actually measured.

## Prompt engineering

The prompt in `src/prompts.py` instructs the model to use only retrieved context,
avoid invented facts and numbers, abstain when evidence is missing, answer in fewer
than 100 words, and cite relevant pages.

## Test

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

For a dependency-light wiring check:

```bash
python scripts/self_check.py
```

## Deploy to Hugging Face Spaces

1. Create a new **Docker Space**.
2. Upload or push this repository.
3. The included Dockerfile exposes port 7860.
4. Keep `GENERATOR_BACKEND=extractive` for reliable free CPU hosting.
5. Enable `transformers` only after confirming the Space has enough RAM.

PDFs are processed in memory and are not saved by the application.

## Limitations

- Image-only/scanned PDFs require OCR and are intentionally out of scope.
- The compact LLM may be weaker than larger hosted models.
- Retrieval does not guarantee that every generated answer is correct.
- Users should verify important answers against the uploaded document.

## Resume bullets after completion

- Developed a Streamlit PDF question-answering application using PyMuPDF, FAISS,
  MiniLM embeddings, and an open-source 0.5B-parameter LLM.
- Fine-tuned the pretrained model on **[actual count]** reviewed examples using
  QLoRA, PEFT, LoRA adapters, and 4-bit NF4 quantization.
- Engineered a top-3 RAG pipeline with document cleaning, overlapping chunks,
  semantic retrieval, prompt grounding, abstention, and page citations.
- Containerized and deployed the application using Docker and Hugging Face Spaces,
  with automated tests through GitHub Actions.

## License

Project code is MIT licensed. Uploaded documents retain their original licences.

