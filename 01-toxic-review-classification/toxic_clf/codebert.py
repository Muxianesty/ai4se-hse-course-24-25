import datasets
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

TEST_SIZE_FRACTION = 0.1
OUTPUT_DIR = Path(f'{os.path.abspath(__file__)}/../../bert-model')

def classifier(dataset: datasets.Dataset):
    train_and_test = dataset.train_test_split(test_size=TEST_SIZE_FRACTION)
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")
    
    options = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=1e-4,
        num_train_epochs=5,
    )
    trainer = Trainer(
        model=model,
        args=options,
        train_dataset=train_and_test["train"],
        eval_dataset=train_and_test["test"],
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.evaluate()
