"""Cluster the patient testimonies and save one cluster per patient.
    python -m scripts.cluster data/raw/unsupervised/Student_Dataset.csv clusters.csv
"""

import sys

from src.unsupervised.cluster import fit_kmeans
from src.unsupervised.loader import read_testimonies
from src.unsupervised.preprocess import analyze_corpus
from src.unsupervised.vectorize import build_reducer, build_vectorizer

N_CLUSTERS = 12


def main():
    if len(sys.argv) != 3:
        print("usage: python -m scripts.cluster INPUT.csv OUTPUT.csv", file=sys.stderr)
        return 84

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        testimonies = read_testimonies(input_path)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 84

    tokens = analyze_corpus(testimonies["testimony"].tolist())
    matrix = build_vectorizer().fit_transform(tokens)
    reduced = build_reducer().fit_transform(matrix)

    testimonies["cluster"] = fit_kmeans(reduced, N_CLUSTERS)
    testimonies[["id", "color", "cluster"]].to_csv(output_path, index=False)

    print(
        f"{len(testimonies)} testimonies sorted into {N_CLUSTERS} clusters, saved to {output_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
