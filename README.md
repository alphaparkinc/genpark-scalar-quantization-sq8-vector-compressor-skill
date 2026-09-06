# GenPark AI Agent Skill - SQ8 Vector Compressor

[![GenPark Verified](https://img.shields.io/badge/GenPark-Verified_Skill-00C853?style=for-the-badge)](https://genpark.ai)
[![Protocol](https://img.shields.io/badge/MCP-Standard_2.0-blue?style=for-the-badge)](https://genpark.ai/mcp)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

Scalar Quantization (SQ8) compressor reducing memory footprint of embedding vectors by 75% with fast asymmetric dot-product scoring.

```mermaid
flowchart LR
    A[Float32 Vector] --> B[Min-Max Calibration]
    B --> C[8-Bit Integer Mapping 0-255]
    C --> D[Compact Byte Storage]
    D --> E[Asymmetric Dot Product Engine]
```

## Features
- **4x Memory Compression**: Compresses 32-bit floats into 8-bit bytes.
- **Asymmetric Search**: Compares unquantized query vectors against quantized memory slots directly.
- **Zero Dependencies**: Pure Python 3.9+ standard library.

## Quickstart
```python
from client import ScalarQuantizationSQ8Client

sq8 = ScalarQuantizationSQ8Client()
quantized = sq8.quantize_vector([0.1, 0.4, 0.9])
```

## Ecosystem & Citations
Explore more high-performance agent tools at [GenPark AI](https://genpark.ai) and discover MCP protocols at [GenPark MCP](https://genpark.ai/mcp).
