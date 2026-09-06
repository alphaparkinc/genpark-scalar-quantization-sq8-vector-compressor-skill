"""
Demonstration of genpark-scalar-quantization-sq8-vector-compressor-skill
"""

from client import ScalarQuantizationSQ8Client

def main():
    sq8 = ScalarQuantizationSQ8Client()

    original_embedding = [0.1234, -0.5678, 0.8901, -0.2345, 0.6789]
    print("Original float32 embedding:", original_embedding)

    compressed = sq8.quantize_vector(original_embedding)
    print("Quantized uint8 vector (4x smaller):", compressed["uint8_vector"])
    print(f"Calibration Scale: min={compressed['min_val']}, max={compressed['max_val']}")

    decompressed = sq8.dequantize_vector(compressed["uint8_vector"], compressed["min_val"], compressed["max_val"])
    print("Reconstructed float vector:", decompressed)

    query = [0.1, -0.5, 0.8, -0.2, 0.6]
    dot_sim = sq8.asymmetric_dot_product(query, compressed)
    print("Asymmetric Dot Product:", dot_sim)

if __name__ == "__main__":
    main()
