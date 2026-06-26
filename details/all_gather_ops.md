# All-Gather Operations

All-Gather is a collective communication primitive where each process in a distributed group sends its portion of a tensor to all other processes, resulting in every process obtaining the complete combined tensor.

## Operation Diagram

```mermaid
flowchart LR
    subgraph Local Data (Before)
        GPU0[GPU 0: holds A]
        GPU1[GPU 1: holds B]
        GPU2[GPU 2: holds C]
    end
    
    GPU0 & GPU1 & GPU2 --> GatherNetwork((All-Gather Collective))
    
    GatherNetwork --> Out0[GPU 0: holds A, B, C]
    GatherNetwork --> Out1[GPU 1: holds A, B, C]
    GatherNetwork --> Out2[GPU 2: holds A, B, C]
```

## How It Works

* **Tensor Parallelism**: Used to reconstruct full activation layers or weight states from individual device shards.
* **ZeRO / Parameter Sharding**: In Zero Redundancy Optimizer (ZeRO) configurations, parameters are gathered via All-Gather right before a layer's forward or backward pass, then immediately freed to save memory.
* **Sequence Parallelism**: All-Gather is used to reconstruct the full sequence dimension from individual sequence shards before calling operations that cannot be easily split (such as softmax or attention projections).

[← Back to README](../README.md)
