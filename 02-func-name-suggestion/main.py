import argparse
from funccraft.data import load_dataset, prepare, save_dataset
from funccraft.models import predict
from pathlib import Path


def main():
    args = parse_args()
    args.func(args)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    default_data_path = Path('./prepared-dataset')
    prepare_data_parser = subparsers.add_parser('prepare-data')
    prepare_data_parser.set_defaults(func=prepare_data)
    prepare_data_parser.add_argument(
        '-o',
        '--output',
        help='Path to save prepared dataset to',
        type=Path,
        default=default_data_path,
    )

    prepare_data_parser.add_argument(
        '-l',
        '--lang',
        help='Dataset code language',
        choices=['python', 'go'],
        default='python',
    )

    predict_parser = subparsers.add_parser('predict-names')
    predict_parser.set_defaults(func=predict_names)
    predict_parser.add_argument(
        '-d',
        '--dataset',
        help='Path to prepared dataset',
        type=Path,
        default=default_data_path,
    )
    predict_parser.add_argument(
        '-m',
        '--model',
        default='Salesforce/codet5p-220m',
    )

    predict_parser.add_argument(
        '-f',
        '--full',
        help='Whether to use full function body for prediction (with comments)',
        action='store_true'
    )

    return parser.parse_args()


def prepare_data(args):
    dataset = prepare(args.lang)
    save_dataset(dataset, args.output)


def predict_names(args):
    dataset = load_dataset(args.dataset)
    predict(dataset, args.model, args.full)


if __name__ == '__main__':
    main()
