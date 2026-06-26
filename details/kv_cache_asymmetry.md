# The KV Cache Memory Asymmetry

With the shift from Multi-Head Attention (MHA) to Grouped-Query Attention (GQA), the number of Key-Value (KV) heads has become much smaller than the number of Query heads. This asymmetry creates partitioning challenges in Tensor Parallelism.

## GQA Sharding Architecture

```mermaid
flowchart TD
    subgraph Query Heads (32 total)
        Q1[Q1..Q8] --> GPU0[GPU 0]
        Q2[Q9..Q16] --> GPU1[GPU 1]
        Q3[Q17..Q24] --> GPU2[GPU 2]
        Q4[Q25..Q32] --> GPU3[GPU 3]
    end
    
    subgraph KV Heads (4 total)
        KV1[KV 1] --> GPU0
        KV2[KV 2] --> GPU1
        KV3[KV 3] --> GPU2
        KV4[KV 4] --> GPU3
    end
```

## How It Works

* In **Grouped-Query Attention (GQA)**, query heads are grouped, and each group shares a single Key-Value head pair.
* If a model has 8 Key-Value heads and we attempt to run it with a Tensor Parallelism (TP) size of 8, each GPU receives exactly 1 Key-Value head.
* However, if we attempt to use a TP size of 16, we would get fractionated heads (0.5 heads per GPU), which is invalid for attention matrix shapes.

## Mitigation

To handle GQA in TP:
1. **Constraint Alignment**: The TP group size must scale as a clean mathematical divisor of the model's native Key-Value head count.
2. **Replication**: When the TP size exceeds the number of KV heads, the KV heads are duplicated across multiple GPUs within the group.

[← Back to README](../README.md)
