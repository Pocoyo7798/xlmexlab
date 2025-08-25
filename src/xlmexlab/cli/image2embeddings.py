import os
import numpy as np
import torch
from PIL import Image
import click
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from xlmexlab.embeddings import EmbeddingExtractor


@click.command()
@click.argument("image_folder", type=str)
@click.argument("output_npy_file", type=str)
@click.option("--filename_list", default=None, help="Optional .txt file to save the list of image filenames")
def image2embeddings(image_folder: str, output_npy_file: str, filename_list: str):
    extractor = EmbeddingExtractor()

    embeddings = []
    filenames = []

    files = sorted(os.listdir(image_folder))
    for fname in files:
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tiff")):
            full_path = os.path.join(image_folder, fname)
            try:
                emb = extractor.extract_embedding(full_path)
                embeddings.append(emb)
                filenames.append(fname)
                print(f"Extracted embedding for {fname}")
            except Exception as e:
                print(f"Failed to extract embedding for {fname}: {e}")


    np.save(output_npy_file, np.stack(embeddings))
    print(f"Embeddings saved to {output_npy_file}")


    if filename_list:
        with open(filename_list, "w") as f:
            for name in filenames:
                f.write(f"{name}\n")
        print(f"Filenames saved to {filename_list}")

def main():
    image2embeddings()


if __name__ == "__main__":
    main()