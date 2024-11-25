from pathlib import Path

import datasets
import python_funccraft

DATASET_SIZE=1000

LANG_MAP = {
    "python": python_funccraft.parseFunc,
}

def prepare(lang: str) -> datasets.Dataset:
    # Implement dataset preparation code here
    dataset = datasets.load_dataset(
        path="code-search-net/code_search_net",
        name="python",
        split="test", trust_remote_code=True
    )

    dataset = dataset.select(range(DATASET_SIZE))
    func_names = [None]*dataset.shape[0]
    func_strings = [None]*dataset.shape[0]
    func_docs = [None]*dataset.shape[0]
    
    lang_func = LANG_MAP[lang]

    for i in range(dataset.shape[0]):
        name_str, body_str, body_stripped_str = lang_func(dataset[i]["whole_func_string"])
        func_names[i] = name_str
        func_strings[i] = body_str
        func_docs[i] = body_stripped_str

    dataset = dataset.add_column("NEW_func_name", func_names)
    dataset = dataset.add_column("NEW_whole_func_string", func_strings)
    dataset = dataset.add_column("NEW_func_no_documentation_string", func_docs)

    return dataset


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))
    selected = dataset.select([9, 19, 21, 62])
    selected.to_json(str(path.joinpath("9-19-21-62.json")))
