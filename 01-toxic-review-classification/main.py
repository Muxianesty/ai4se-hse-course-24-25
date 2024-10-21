import argparse
from pathlib import Path

from toxic_clf.data import load_dataset, prepare, save_dataset
from toxic_clf.models import classifier

def prepare_data(args):
    dataset = prepare(args.input)
    save_dataset(dataset, args.output)

def classify(args):
    dataset = load_dataset(args.dataset)
    classifier(dataset, args.model)

DEFAULT_PREP_DATASET_PATH = Path('./prepared-data')

DEFAULT_IN_DATASET_PATH = Path('./data/models/code-review-dataset-full.xlsx')

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    prepare_data_parser = subparsers.add_parser('prepare-data')
    prepare_data_parser.set_defaults(func=prepare_data)
    prepare_data_parser.add_argument(
        '-i',
        '--input',
        help='Path to load raw dataset',
        type=Path,
        default=DEFAULT_IN_DATASET_PATH,
    )
    prepare_data_parser.add_argument(
        '-o',
        '--output',
        help='Path to save prepared dataset to',
        type=Path,
        default=DEFAULT_PREP_DATASET_PATH,
    )

    predict_parser = subparsers.add_parser('classify')
    predict_parser.set_defaults(func=classify)
    predict_parser.add_argument(
        '-d',
        '--dataset',
        help='Path to prepared dataset',
        type=Path,
        default=DEFAULT_PREP_DATASET_PATH,
    )
    predict_parser.add_argument(
        '-m',
        '--model',
        choices=['classic_log', 'codebert'],
        default='classic_log',
    )

    return parser.parse_args()

def main():
    args = parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
