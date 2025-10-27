# 🖌️ Advanced Smart Manhwa Cleaner

This project is a web-based tool built with Gradio to automatically "clean" manhwa or manga pages. It streamlines the process of removing text and enhancing image quality, making it an essential tool for scanlation groups or anyone looking to prepare raw comic pages.

## ✨ Features

- **Automatic Text Detection**: Utilizes PaddleOCR to accurately identify text boxes on comic pages.
- **High-Quality Text Removal**: Employs the LaMa Inpainting model to intelligently fill in the areas where text has been removed, preserving the original artwork.
- **Image Super-Resolution**: Integrates RealESRGAN to upscale the cleaned images, resulting in high-resolution, print-quality pages.
- **User-Friendly Web Interface**: A simple Gradio interface allows users to easily upload their images and configure the cleaning process.
- **Batch Processing**: Supports processing of entire folders (chapters) at once.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended for better performance)
- `git` for cloning the repository

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/user/advanced-smart-manhwa-cleaner.git
    cd advanced-smart-manhwa-cleaner
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This process may take some time as it will download several large deep learning models.*

### Usage

1.  **Run the Gradio application:**
    ```bash
    python clean_chapter.py
    ```

2.  **Open the web interface:**
    Open your web browser and navigate to the local URL provided by Gradio (usually `http://127.0.0.1:7860`).

3.  **Using the Interface:**
    - **Select chapter images or drag the entire folder**: Upload all the image files for a single chapter, or drag and drop a folder containing multiple chapter subfolders.
    - **Text language in images**: Choose the language of the text on the pages for more accurate detection.
    - **Super-resolution scale**: Select the upscaling factor. `2x` is faster, while `4x` provides higher quality.
    - **Save cleaned images format**: Choose whether to save the output images as `PNG` or `JPG`.

4.  **Processing:**
    After configuring the options, the process will start. You can monitor the progress in the terminal and in the Gradio interface.

5.  **Output:**
    The cleaned images will be saved in a new subfolder named `chapter_cleaned` inside each processed chapter's directory.

## ⚙️ How It Works

The script performs the following steps for each image:

1.  **Text Detection**: It uses `PaddleOCR` to find the bounding boxes of all text elements.
2.  **Mask Creation**: A mask is generated based on the detected text boxes.
3.  **Text Removal**: The `LaMaInpainting` model is used to inpaint the masked areas, effectively removing the text while reconstructing the background.
4.  **Quality Enhancement**: The `RealESRGAN` model upscales the cleaned image to a higher resolution.

Models are cached in memory after the first run to speed up subsequent processing.

## 🤝 Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.
