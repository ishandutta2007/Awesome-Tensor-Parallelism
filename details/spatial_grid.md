# The Multi-Dimensional Spatial Grid Era (~2021–2023)

To overcome the communication and memory scaling limits of 1D tensor parallelism, frameworks like Colossal-AI introduced 2D, 2.5D, and 3D tensor parallelism. These methods slice weight matrices and activations along multiple dimensions concurrently.

## Grid Topology Diagram

```mermaid
flowchart TD
    subgraph 2D Grid Representation (2x2 GPUs)
        GPU_00[GPU 0,0] <-->|Row Sync| GPU_01[GPU 0,1]
        GPU_10[GPU 1,0] <-->|Row Sync| GPU_11[GPU 1,1]
        GPU_00 <-->|Col Sync| GPU_10[GPU 1,0]
        GPU_01 <-->|Col Sync| GPU_11[GPU 1,1]
    end
```

## How It Works

* **2D Tensor Parallelism**: Based on the SUMMA matrix multiplication algorithm, it shards the input activations and weights into a 2-dimensional grid of $R \times C$ processors.
* **2.5D Tensor Parallelism**: Arranges processors in a 3D grid of size $R \times C \times D$, where the third dimension replicates weights or intermediate states to save communication bandwidth.
* **3D Tensor Parallelism**: Slices weight matrices along three dimensions ($X, Y, Z$), achieving optimal theoretical memory distribution but requiring highly complex coordination across the processor grid.

## Key Benefits

* **Memory Footprint**: Reduces activation memory by sharding activations more aggressively than 1D.
* **Scale**: Enables model scaling to hundreds of billions of parameters across commodity nodes without relying exclusively on expensive single-node systems.

[← Back to README](../README.md)
