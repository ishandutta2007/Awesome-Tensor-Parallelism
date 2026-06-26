# The 1D Megatron-LM Baseline Era

Intra-layer model parallelization was popularized by NVIDIA's Megatron-LM framework in 2019. It splits the weight matrices of a Transformer layer across multiple GPUs within a single node to scale models beyond the memory capacity of a single graphics card.

## Architecture Diagram

```mermaid
flowchart TD
    subgraph Column-Parallel Attention (Q, K, V)
        Input[Input Tensor X] --> GPU0_Col[GPU 0: Column Splitting W_Col1]
        Input --> GPU1_Col[GPU 1: Column Splitting W_Col2]
        GPU0_Col --> Out0_Col[Partial Output Y1]
        GPU1_Col --> Out1_Col[Partial Output Y2]
    end
    
    subgraph Row-Parallel Projection
        Out0_Col --> GPU0_Row[GPU 0: Row Splitting W_Row1]
        Out1_Col --> GPU1_Row[GPU 1: Row Splitting W_Row2]
        GPU0_Row --> Sum[All-Reduce Summation]
        GPU1_Row --> Sum
        Sum --> FinalOut[Final Attention Output]
    end
```

## How It Works

Megatron-LM 1D Tensor Parallelism splits the Multi-Layer Perceptron (MLP) and Self-Attention layers:
1. **Self-Attention block**: The Query, Key, and Value ($Q, K, V$) projections are sharded column-wise. The attention output projection ($O$) is sharded row-wise.
2. **MLP block**: The first linear layer (Gated/Up-projection) is sharded column-wise, and the second linear layer (Down-projection) is sharded row-wise.

This specific column-then-row parallel sequence is mathematically equivalent to the standard non-parallelized layers but reduces communication overhead: it only requires one **All-Reduce** operation in the forward pass (at the end of the row-parallel layer) and one in the backward pass.

## Trade-offs

* **Advantages**:
  * Simple to implement without changes to compiler level.
  * Fits well into standard intra-node NVLink architecture.
* **Limitations**:
  * Dual communication synchronization (All-Gather/All-Reduce) per transformer layer, which stalls execution on slow network topologies.

[← Back to README](../README.md)
