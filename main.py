"""Run both packaged examples from one entry point."""

from examples.gmm_recursion_example import run as run_gmm
from examples.ngpca_example import run as run_ngpca


def main() -> None:
    """Fit the NGPCA and GMM examples and print compact summaries."""
    ngpca, _ = run_ngpca()
    gmm, _ = run_gmm()
    print(f"NGPCA centers: {ngpca.centers().shape}")
    print(f"GMM weights: {gmm.weights.round(3)}")


if __name__ == "__main__":
    main()
