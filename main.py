from examples.gmm_recursion_example import run as run_gmm
from examples.ngpca_example import run as run_ngpca


def main():
    ngpca, _ = run_ngpca()
    gmm, _ = run_gmm()
    print(f"NGPCA centers: {ngpca.centers().shape}")
    print(f"GMM weights: {gmm.weights.round(3)}")


if __name__ == "__main__":
    main()
