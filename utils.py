import base64
import gc
import io

import lancedb
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from transformers import (
    pipeline,
    CLIPProcessor,
    CLIPModel
)

# ------------------ Visualization Utilities ------------------ #

def show_base64(base64_string: str) -> None:
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


# ------------------ Image Segmentation Utilities ------------------ #

def crop_image_by_seg(image: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    """
    Apply segmentation mask to the image, masking out everything else to white.
    """
    binary_mask = segmentation.astype(int)
    white_canvas = np.ones_like(image) * 255
    return np.where(binary_mask[..., None] > 0, image, white_canvas)


sam_model = pipeline("mask-generation", model="facebook/sam-vit-huge", device='cpu')

def get_cropped_segments(image: np.ndarray) -> list[np.ndarray]:
    """
    Generate cropped segments from an image using SAM.
    """
    outputs = sam_model(image, points_per_batch=64)
    masks = outputs["masks"]
    return [crop_image_by_seg(image, mask) for mask in masks]


# ------------------ CLIP Embedding Utilities ------------------ #

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)


def get_embeddings(image: np.ndarray, model=clip_model, processor=clip_processor) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)
    return image_embeddings.flatten().cpu().numpy()


def get_embeddings_text(text: str, model=clip_model, processor=clip_processor) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs)
    return text_embeddings.flatten().cpu().numpy()


# ------------------ LanceDB Utilities ------------------ #

def create_segment_db(raw_image: np.ndarray, db_path: str = "Segmentation_Table") -> lancedb.db.LanceTable:
    """
    Segments the image, generates embeddings, stores them with image content in LanceDB.
    """
    segments = get_cropped_segments(raw_image)
    segment_embeddings = [get_embeddings(seg) for seg in segments]

    db = lancedb.connect(db_path)
    schema = pa.schema([
        pa.field("embedding", pa.list_(pa.float32(), list_size=512)),
        pa.field("content", pa.string())
    ])
    table = db.create_table(db_path, schema=schema, mode='overwrite')

    records = []
    for emb, seg in zip(segment_embeddings, segments):
        img = Image.fromarray(seg.astype('uint8'), 'RGB')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        records.append({"embedding": emb, "content": img_base64})

    table.add(records)
    return table
