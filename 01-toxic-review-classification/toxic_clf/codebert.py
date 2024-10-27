import datasets
import evaluate
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

TEST_SIZE_FRACTION = 0.1
OUTPUT_DIR = Path(f'{os.path.abspath(__file__)}/../../bert-model')

conf_matrix_metric = evaluate.load("confusion_matrix")
f1_metric = evaluate.load("f1")

def classifier(dataset: datasets.Dataset):
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")
    
    def tokenization_map(data):
        return tokenizer(data["message"], truncation=True, padding=True)
    mapped_dataset = dataset.map(tokenization_map, batched=True)

    mapped_dataset = mapped_dataset.rename_column("is_toxic", "labels")
    train_and_test = mapped_dataset.train_test_split(test_size=TEST_SIZE_FRACTION)

    def compute_metrics(results):
        y_pred, y = results
        y_pred = y_pred.argmax(axis=1)
        f1 = f1_metric.compute(predictions=y_pred, references=y, average='weighted')
        conf_matrix = conf_matrix_metric.compute(predictions=y_pred, references=y)
        return {'f1': f1['f1'], "conf_matrix": conf_matrix["confusion_matrix"]}

    options = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        remove_unused_columns=True,
    )
    trainer = Trainer(
        model=model,
        args=options,
        train_dataset=train_and_test["train"],
        eval_dataset=train_and_test["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    stats = trainer.evaluate()
    print(stats)
