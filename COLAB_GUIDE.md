# Fine-tuning in Google Colab

1. Push this project to GitHub.
2. Open Colab and select **Runtime → Change runtime type → T4 GPU**.
3. Run:

```bash
git clone https://github.com/YOUR_USERNAME/DocuChat-AI.git
cd DocuChat-AI
pip install -r requirements-train.txt
python scripts/train_qlora.py --dataset data/training_data.jsonl --epochs 2
```

The included five examples only verify the data format. Expand and manually review
100–300 examples before reporting this as a trained project. Download the generated
`models/docuchat-lora` folder when training finishes.

If GPU memory is insufficient, change the training batch size from 4 to 2 or 1.

