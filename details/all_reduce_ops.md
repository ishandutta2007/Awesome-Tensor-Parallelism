# All-Reduce Operations

All-Reduce is a key collective communication primitive used in distributed training. It combines data (typically by summing) from all processes and distributes the combined result back to all processes.

## Operation Diagram

```mermaid
flowchart LR
    subgraph Local States (Before)
        GPU0[GPU 0: holds A]
        GPU1[GPU 1: holds B]
        GPU2[GPU 2: holds C]
    end
    
    GPU0 & GPU1 & GPU2 --> RingAllReduce((Ring All-Reduce / NVLink))
    
    RingAllReduce --> Out0[GPU 0: holds A+B+C]
    RingAllReduce --> Out1[GPU 1: holds A+B+C]
    RingAllReduce --> Out2[GPU 2: holds A+B+C]
```

## How It Works

In 1D Tensor Parallelism, All-Reduce is required at the terminal boundary of Row-Parallel linear layers:
1. Each GPU holds a partial slice ($Y_i$) of the matrix multiplication output.
2. The All-Reduce operation coordinates across the inter-GPU network to sum these slices.
3. Once completed, every GPU in the TP group contains the exact identical combined output tensor ($Y = \sum Y_i$).

## Performance Impact

Because All-Reduce blocks the forward pass until all nodes have synchronized, it is a primary driver of latency in distributed clusters. Optimizing this via high-speed topologies like NVLink (intra-node) or InfiniBand (inter-node) is critical.

[← Back to README](../README.md)
