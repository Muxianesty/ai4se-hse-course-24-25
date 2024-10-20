from pathlib import Path

import datasets
import pandas as pd


def prepare_dataset(dataset: pd.DataFrame):
    dataset = dataset.dropna().drop_duplicates()
    return dataset

def prepare(raw_data: Path) -> datasets.Dataset:
    pd_dataset = pd.read_excel(raw_data)
    return datasets.Dataset.from_pandas(prepare_dataset(pd_dataset))

def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))
