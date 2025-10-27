import os
import cv2
import numpy as np
from PIL import Image
import torch
import zipfile
import tempfile
from lama_cleaner.model import LaMaInpainting
from realesrgan import RealESRGAN
from paddleocr import PaddleOCR
import gradio as gr

# =========================
# 🔧 Device Setup
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Using device: {device}")

# =========================
# Load Inpainting Model (Static)
# =========================
# This model is loaded only once because it does not depend on user settings
lama_model = LaMaInpainting(model_type='lama', device=device)
print("✅ Text removal model loaded")

# Cache for models that depend on user settings
model_cache = {}

def get_ocr_model(lang):
    """Load or return OCR model from cache"""
    if lang not in model_cache:
        print(f"🔄 Loading OCR model for language: {lang}")
        model_cache[lang] = PaddleOCR(use_angle_cls=True, lang=lang)
    return model_cache[lang]

def get_realesrgan_model(scale):
    """Load or return super-resolution model from cache"""
    key = f"realesrgan_x{scale}"
    if key not in model_cache:
        print(f"🔄 Loading super-resolution model with scale: {scale}x")
        model = RealESRGAN(device, scale=scale)
        # Weights file depends on scale
        model.load_weights(f'RealESRGAN_x{scale}plus.pth', download=True)
        model_cache[key] = model
    return model_cache[key]

# =========================
# Single Page Processing Function
# =========================
def clean_page(image, ocr_model, lama_model, realesrgan_model):
    """Clean a single page: detect text, remove it, then enhance quality"""
    image_np = np.array(image.convert('RGB'))
    h, w = image_np.shape[:2]

    # 1. Text detection
    result = ocr_model.ocr(image_np)
    
    # 2. Create text mask
    mask = np.zeros((h, w), dtype=np.uint8)
    if result and result[0]:
        for line in result[0]:
            pts = np.array(line[0], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

    # 3. Remove text
    cleaned_image = lama_model(image_np, mask)

    # 4. Enhance quality
    upscaled_image = realesrgan_model.predict(cleaned_image)

    return Image.fromarray(upscaled_image)

# =========================
# Full Chapter Processing Function
# =========================
def clean_chapter_folder(chapter_folder, ocr_model, lama_model, realesrgan_model, output_format):
    """Process all images in a single chapter folder"""
    output_folder = os.path.join(chapter_folder, "chapter_cleaned")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp')
    files = sorted([f for f in os.listdir(chapter_folder) if f.lower().endswith(supported_formats)])
    
    if not files:
        print(f"⚠️ No images found in folder: {chapter_folder}")
        return None

    for idx, file in enumerate(files, 1):
        input_path = os.path.join(chapter_folder, file)
        print(f"📄 Processing page {idx}/{len(files)}: {file}")
        
        try:
            image = Image.open(input_path)
            result_image = clean_page(image, ocr_model, lama_model, realesrgan_model)
            
            # Save in the desired format
            file_name, _ = os.path.splitext(file)
            output_filename = f"{file_name}.{output_format.lower()}"
            output_path = os.path.join(output_folder, output_filename)

            if output_format == 'JPG':
                result_image.convert('RGB').save(output_path, 'jpeg', quality=95)
            else:
                result_image.save(output_path, output_format.upper())

        except Exception as e:
            print(f"❌ Error processing file {file}: {e}")

    return output_folder

# =========================
# Main Function for Gradio
# =========================
def process_all_chapters(uploaded_files, lang, scale, output_format, progress=gr.Progress(track_tqdm=True)):
    """Function connecting UI with backend logic"""
    if not uploaded_files:
        return None, "Please select image files first."

    # Load models based on user selection
    progress(0, desc="Loading models...")
    ocr_model = get_ocr_model(lang)
    realesrgan_model = get_realesrgan_model(scale)

    # Infer root folder from uploaded files
    all_paths = [f.name for f in uploaded_files]
    root_path = os.path.commonpath(all_paths) if len(all_paths) > 1 else os.path.dirname(all_paths[0])

    # Search for chapter folders
    potential_chapters = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

    chapters_to_process = []
    if potential_chapters:
        chapters_to_process = [os.path.join(root_path, d) for d in potential_chapters]
        print(f"📂 Found {len(chapters_to_process)} chapters in: {root_path}")
    else:
        chapters_to_process = [root_path]
        print(f"📂 No chapters found, processing current folder: {root_path}")

    if not chapters_to_process:
        return None, "No images found for processing in the selected folder."

    processed_folders = []
    for chapter_path in progress.tqdm(chapters_to_process, desc="Processing chapters"):
        output_folder = clean_chapter_folder(chapter_path, ocr_model, lama_model, realesrgan_model, output_format)
        if output_folder:
            processed_folders.append(output_folder)

    if not processed_folders:
        return None, "No images were processed."

    # Create a zip file with the results
    zip_path = os.path.join(tempfile.gettempdir(), "cleaned_chapters.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for folder in processed_folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(folder))
                    zipf.write(file_path, arcname=arcname)

    return zip_path, f"✅ Processing complete. Your download is ready."

# =========================
# Gradio Interface
# =========================
interface = gr.Interface(
    fn=process_all_chapters,
    inputs=[
        gr.Files(label="Select chapter images or drag the entire folder"),
        gr.Dropdown(['en', 'ar', 'ja', 'ko', 'ch_sim', 'fr', 'de'], label="Text language in images", value='en', info="Choose the language for accurate removal"),
        gr.Dropdown([2, 4], label="Super-resolution scale", value=2, info="2x is faster, 4x gives higher quality"),
        gr.Radio(['PNG', 'JPG'], label="Save cleaned images format", value='PNG')
    ],
    outputs=[
        gr.File(label="Download cleaned chapters (.zip)"),
        gr.Textbox(label="Status", lines=5)
    ],
    title="🖌️ Advanced Smart Manhwa Cleaner",
    description="Upload the images you want to clean. You can upload a single chapter or drag a folder containing multiple chapter folders.",
    allow_flagging='never',
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # Note: On first run, models will be loaded and cached. This may take some time.
    # For RealESRGAN scale 4, make sure you have 'RealESRGAN_x4plus.pth' file available or let it download.
    interface.launch(server_name="0.0.0.0")
