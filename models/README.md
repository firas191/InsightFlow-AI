# Models directory

This is where the **fine-tuned ticket classifier** lives after you train it on Kaggle.

## How to populate it

1. Run `notebooks/train_ticket_classifier.ipynb` on Kaggle (GPU accelerator).
2. The last cell writes `insightflow_ticket_classifier.zip` to the Kaggle output.
3. Download it and unzip here so you end up with:

```
models/
└── insightflow_ticket_classifier/
    ├── config.json              # encoder config (HF)
    ├── model.safetensors        # fine-tuned encoder weights
    ├── tokenizer.json / vocab   # tokenizer files
    ├── heads.pt                 # the four classification heads
    └── label_maps.json          # task -> label list (index order)
```

The path is set in `config/config.yaml` under `models.classifier_dir`.

## What happens if it's empty

The agent still runs. `src/classifier/inference.py` detects the missing
`label_maps.json` and automatically falls back to a **zero-shot** classifier
(`facebook/bart-large-mnli`). Slower and less accurate, but the whole pipeline
works end-to-end with zero training. Drop the trained model in and the agent
switches to `finetuned` mode on the next run.

> Model weights are git-ignored (see `.gitignore`) — they're too big for the repo.
