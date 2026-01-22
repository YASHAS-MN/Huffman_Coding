from PIL import Image
import numpy as np
from collections import Counter
import heapq
from bitarray import bitarray
import pickle

# ──────────────────────────────────────
# Huffman Node Class
# ──────────────────────────────────────
class Node:
    def __init__(self, pixel, freq):
        self.pixel = pixel
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# ──────────────────────────────────────
# Load and Flatten Image
# ──────────────────────────────────────
def load_image(path):
    img = Image.open(path).convert('L')  # Grayscale
    img_data = np.array(img)
    flat_pixels = img_data.flatten()
    return flat_pixels, img.size, img.mode

# ──────────────────────────────────────
# Build Huffman Tree and Generate Codes
# ──────────────────────────────────────
def build_huffman_tree(pixels):
    freq_map = Counter(pixels)
    heap = [Node(pixel, freq) for pixel, freq in freq_map.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def generate_codes(node, current_code="", codes=None):
    if codes is None:
        codes = {}
    if node is None:
        return codes
    if node.pixel is not None:
        codes[node.pixel] = current_code
        return codes
    generate_codes(node.left, current_code + "0", codes)
    generate_codes(node.right, current_code + "1", codes)
    return codes

# ──────────────────────────────────────
# Encode and Save Compressed Data
# ──────────────────────────────────────
def encode_pixels(pixels, huffman_codes):
    encoded_bits = bitarray()
    for pixel in pixels:
        encoded_bits.extend(huffman_codes[pixel])
    return encoded_bits

def save_compressed(encoded_bits, huffman_codes, size, filename="compressed.huff"):
    with open(filename, "wb") as f:
        data = {
            "size": size,
            "codes": huffman_codes,
            "data": encoded_bits
        }
        pickle.dump(data, f)

# ──────────────────────────────────────
# Decode and Reconstruct Image
# ──────────────────────────────────────
def decode_pixels(encoded_bits, codes, size):
    reversed_codes = {v: k for k, v in codes.items()}
    current_code = ""
    decoded_pixels = []
    for bit in encoded_bits.to01():
        current_code += bit
        if current_code in reversed_codes:
            decoded_pixels.append(reversed_codes[current_code])
            current_code = ""
    return np.array(decoded_pixels, dtype=np.uint8).reshape(size[1], size[0])

def decompress(filename="compressed.huff", output_image="decompressed.png"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    size = data["size"]
    codes = data["codes"]
    encoded_bits = data["data"]
    pixels_array = decode_pixels(encoded_bits, codes, size)
    img = Image.fromarray(pixels_array, mode='L')
    img.save(output_image)

# ──────────────────────────────────────
# Main Program
# ──────────────────────────────────────
if __name__ == "__main__":
    print("\U0001F4A1 Main script is running.")
    try:
        pixels, size, mode = load_image("test.bmp")
        print("\u2705 Image loaded:", size)

        root = build_huffman_tree(pixels)
        print("\u2705 Tree built")

        huffman_codes = generate_codes(root)
        print("\u2705 Codes generated:", len(huffman_codes))

        encoded_bits = encode_pixels(pixels, huffman_codes)
        print("\u2705 Pixels encoded. Total bits:", len(encoded_bits))

        save_compressed(encoded_bits, huffman_codes, size)
        print("\u2705 Compressed")

        decompress()
        print("\u2705 Decompressed")

    except Exception as e:
        print("\u274C Error occurred:", e)

import os

def get_file_size_kb(filename):
    return os.path.getsize(filename) / 1024

original_size = get_file_size_kb("test.bmp")
compressed_size = get_file_size_kb("compressed.huff")
decompressed_size = get_file_size_kb("decompressed.png")

compression_ratio = original_size / compressed_size

print(f"\n📦 Original size:     {original_size:.2f} KB")
print(f"🗜️  Compressed size:  {compressed_size:.2f} KB")
print(f"🧩 Decompressed size: {decompressed_size:.2f} KB")
print(f"⚖️  Compression Ratio: {compression_ratio:.2f} : 1")

# Compare original and decompressed pixel-wise
original_pixels, _, _ = load_image("test.bmp")
restored_pixels, _, _ = load_image("decompressed.png")

if np.array_equal(original_pixels, restored_pixels):
    print("✅ Decompressed image is IDENTICAL to original.")
else:
    print("❌ Images differ!")
