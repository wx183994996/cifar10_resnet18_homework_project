import argparse
from utils.plot import plot_curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    plot_curves(args.csv, args.out_dir)
    print(f"curves saved to {args.out_dir}")


if __name__ == "__main__":
    main()
