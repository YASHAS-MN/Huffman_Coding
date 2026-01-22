import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
import heapq
from bitarray import bitarray
import pickle
import os

# Huffman Node
class Node:
    def __init__(self, pixel, freq):
        self.pixel = pixel
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# Build Huffman Tree
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

# Generate Huffman Codes
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

# Encode pixel array
def encode_pixels(pixels, huffman_codes):
    encoded_bits = bitarray()
    for pixel in pixels:
        encoded_bits.extend(huffman_codes[pixel])
    return encoded_bits

# Decode bitstream
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

# Compress Image
def compress_image(image):
    img = image.convert("L")
    img_data = np.array(img)
    flat_pixels = img_data.flatten()
    size = img.size

    root = build_huffman_tree(flat_pixels)
    huffman_codes = generate_codes(root)
    encoded_bits = encode_pixels(flat_pixels, huffman_codes)

    data = {
        "size": size,
        "codes": huffman_codes,
        "data": encoded_bits
    }

    with open("compressed.huff", "wb") as f:
        pickle.dump(data, f)

    return len(flat_pixels), len(encoded_bits), size

# Decompress Image
def decompress_image():
    with open("compressed.huff", "rb") as f:
        data = pickle.load(f)

    size = data["size"]
    codes = data["codes"]
    encoded_bits = data["data"]

    pixels_array = decode_pixels(encoded_bits, codes, size)
    img = Image.fromarray(pixels_array, mode='L')
    img.save("decompressed.png")
    return img

# Get file size in KB
def get_file_size_kb(filepath):
    return os.path.getsize(filepath) / 1024

# Streamlit UI
st.set_page_config(page_title="Image Compressor with Huffman Coding", layout="centered")
st.title("🗜️ Huffman Image Compressor")
st.markdown("Upload an image (preferably grayscale BMP/PNG) and compress it using Huffman coding.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "bmp", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Compress Image"):
        original_pixels, encoded_bits, size = compress_image(image)
        original_size_kb = get_file_size_kb(uploaded_file.name)
        compressed_size_kb = get_file_size_kb("compressed.huff")

        st.success("Image compressed successfully!")
        st.write(f"📏 Original Pixels: {original_pixels}")
        st.write(f"🔢 Encoded Bits: {encoded_bits}")
        st.write(f"📦 Original Size: {original_size_kb:.2f} KB")
        st.write(f"🗜️ Compressed Size: {compressed_size_kb:.2f} KB")
        st.write(f"⚖️ Compression Ratio: {original_size_kb / compressed_size_kb:.2f} : 1")

    if st.button("Decompress Image"):
        output_img = decompress_image()
        st.image(output_img, caption="Decompressed Image", use_column_width=True)
        st.success("Image decompressed successfully!")
