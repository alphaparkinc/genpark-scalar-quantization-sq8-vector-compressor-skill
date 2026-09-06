"""
Uniform 8-Bit Scalar Quantization (SQ8) Vector Compressor.
Zero external dependencies, standard library only.
"""

from typing import Dict, List, Any, Optional, Tuple

class ScalarQuantizationSQ8Client:
    """
    Compresses float32 vectors to uint8 (0-255) representations (4x RAM reduction):
    - Computes global/per-vector min-max scale factors
    - Quantizes float32 -> uint8
    - Dequantizes uint8 -> float32
    - Calculates asymmetric dot-product distance without full decompression
    """

    def __init__(self):
        pass

    def quantize_vector(self, vector: List[float]) -> Dict[str, Any]:
        """Quantizes float vector to 8-bit integers [0, 255]."""
        if not vector:
            return {"uint8_vector": [], "min_val": 0.0, "max_val": 0.0}

        min_val = min(vector)
        max_val = max(vector)
        range_val = max_val - min_val if max_val != min_val else 1.0

        uint8_data = []
        for val in vector:
            normalized = (val - min_val) / range_val
            int_val = int(round(normalized * 255.0))
            uint8_data.append(max(0, min(255, int_val)))

        return {
            "uint8_vector": uint8_data,
            "min_val": min_val,
            "max_val": max_val,
            "dimensions": len(vector)
        }

    def dequantize_vector(self, uint8_vector: List[int], min_val: float, max_val: float) -> List[float]:
        """Reconstructs approximate float vector from uint8 representation."""
        range_val = max_val - min_val
        return [round(min_val + (int_val / 255.0) * range_val, 6) for int_val in uint8_vector]

    def asymmetric_dot_product(self, float_query: List[float], quantized_target: Dict[str, Any]) -> float:
        """Calculates dot product between float query and SQ8 quantized target."""
        u8_vec = quantized_target["uint8_vector"]
        min_v = quantized_target["min_val"]
        max_v = quantized_target["max_val"]
        range_v = max_v - min_v

        dot = 0.0
        for q_val, u_val in zip(float_query, u8_vec):
            t_approx = min_v + (u_val / 255.0) * range_v
            dot += q_val * t_approx

        return round(dot, 6)
