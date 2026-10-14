# Resformer
# Genomic Prediction with Genotype by Environment Interaction Using a Residual Network and Transformer Framework

This project uses PyTorch to predict crop yield from genotype and
environmental features. The model combines gated 1D convolutions, a
Transformer encoder, an environmental encoder, bidirectional multi-head
cross-attention, and residual feature fusion.

## Requirements

-   Python 3.9 or compatible version
-   PyTorch, pandas, NumPy, scikit-learn, SciPy

Install dependencies:

``` bash
pip install torch pandas numpy scikit-learn scipy
```

## Input Data

Update the file paths in `main()` before running the script.

-   **Genotype CSV**: first column contains hybrid IDs (used as the
    index); remaining columns are numeric genotype features.
-   **Phenotype CSV**: the columns must conain `Environment`,
    `Hybrid`, and `Yield` in that order.
-   **Environment CSV**: first column contains environment IDs;
    remaining columns are numeric environmental variables.

## Configuration

Main settings include batch size 32, up to 200 epochs, learning rate
0.0005, hidden dimension 64, 8 attention heads, and early-stopping
patience 50. 

