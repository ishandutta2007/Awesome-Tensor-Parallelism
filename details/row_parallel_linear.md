# Row-Parallel Linear Layers

Row-parallel linear layers are the second core tensor decomposition building block of 1D tensor parallelism. They split a linear layer's weight matrix horizontally across multiple devices.

## Computation Flow Diagram

```mermaid
flowchart TD
    subgraph Sharded Inputs
        X1[Input Slice X1]
        X2[Input Slice X2]
    end
    X1 --> GPU0[GPU 0: computes X1 * W_1]
    X2 --> GPU1[GPU 1: computes X2 * W_2]
    GPU0 --> Y1[Partial Sum Y1]
    GPU1 --> Y2[Partial Sum Y2]
    Y1 & Y2 --> Add[All-Reduce Summation]
    Add --> Y[Combined Output Y = Y1 + Y2]
```

## How It Works

For a linear layer $Y = XW$, row-parallel splitting divides the weight matrix $W$ along its rows across $N$ devices:

$$W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix}$$

To perform mathematically valid matrix multiplication, the input tensor $X$ must also be split column-wise across the devices:

$$X = [X_1, X_2, \dots, X_N]$$

Each device performs local matrix multiplication:

$$Y_i = X_i W_i$$

To obtain the final correct output tensor, the partial products must be summed across all GPUs. This is achieved using an **All-Reduce** operation:

$$Y = \sum_{i=1}^N Y_i$$

## Applications

* **Self-Attention Output Projection ($O$)**: Merges heads back together and performs the final linear projection.
* **MLP Down-projection layer**: Slices down-projection weights horizontally and aggregates output shards.

[← Back to README](../README.md)
