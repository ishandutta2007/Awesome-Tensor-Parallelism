# The NVLink Hard Boundary Constraint

Tensor Parallelism (TP) splits weight matrices and requires synchronization within a single transformer layer. This creates a high communication frequency, making execution highly sensitive to network bandwidth constraints.

## Network Topology Diagram

```mermaid
flowchart TD
    subgraph Single Chassis (Node)
        GPU0[GPU 0] <-->|NVLink: 900 GB/s| GPU1[GPU 1]
        GPU1 <-->|NVLink: 900 GB/s| GPU2[GPU 2]
    end
    
    subgraph Inter-Node Network
        Node1[Node 1] <-->|InfiniBand: 50 GB/s| Node2[Node 2]
    end
```

## How It Works

Because TP splits matrices at the matrix multiplication level, collective communication operations (All-Reduce / All-Gather) must execute twice per transformer layer. 
* **Intra-node (NVLink)**: Bandwidth is typically very high (e.g., up to 900 GB/s per GPU on NVIDIA H100 systems), meaning communication takes only a fraction of a millisecond.
* **Inter-node (InfiniBand/Ethernet)**: Bandwidth is significantly lower (e.g., 50–100 GB/s), leading to extreme communication bottlenecks if TP is split across physical chassis nodes.

## Production Best Practices

To prevent network bottlenecks from stalling GPU tensor cores, production systems restrict the maximum Tensor Parallelism group size to the boundary of a single node (typically **TP size of 4 or 8**). For scaling models across multiple nodes, TP is combined with Pipeline Parallelism (PP) or Data Parallelism (DP) which communicate less frequently.

[← Back to README](../README.md)
