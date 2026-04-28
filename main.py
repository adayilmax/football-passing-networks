"""
Graph Representation Learning on Relational Sports Data
========================================================
GCN and GAT with Interpretable Attention for Match Outcome Prediction
"""


def main() -> None:
    print("""
Graph Representation Learning on Relational Sports Data
========================================================
GCN and GAT with Interpretable Attention for Match Outcome Prediction

Encoding team passing behaviour as weighted directed graphs and learning
over relational structure with GCN and GAT — does topology carry signal
beyond aggregate statistics?

Run the pipeline scripts in order:

  uv run python scripts/build_dataset.py                # Build PyG graph dataset from StatsBomb data
  uv run python scripts/train_models.py                 # Train GCN and GAT models
  uv run python scripts/evaluate_models.py              # Evaluate and generate result plots
  uv run python scripts/visualize_interpretability.py   # GAT attention + t-SNE
  uv run python scripts/visualize_explainability.py     # GNNExplainer + agreement analysis
  uv run python scripts/compare_temporal.py             # Full-match vs first-half analysis

See README.md for full documentation.
""")


if __name__ == "__main__":
    main()
