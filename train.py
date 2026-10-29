"""Optional compatibility CLI. Prefer train_EfficientNetB0.py etc."""
import argparse
from heritagevision.models import SUPPORTED_MODELS
from heritagevision.training import train_cli


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--model', choices=SUPPORTED_MODELS, default='efficientnet_b0')
    args, rest = p.parse_known_args()
    train_cli(args.model, argv=rest)


if __name__ == '__main__':
    main()
