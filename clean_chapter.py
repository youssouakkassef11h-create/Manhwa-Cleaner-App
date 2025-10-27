import os
import cv2
import numpy as np
from PIL import Image
import torch
from lama_cleaner.model import LaMaInpainting
from realesrgan import RealESRGAN
from paddleocr import PaddleOCR
import gradio as gr

# =========================
# 🔧 إعداد الجهاز
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ استخدام الجهاز: {device}")

# =========================
# تحميل نموذج الإزالة (ثابت)
# =========================
# يتم تحميل هذا النموذج مرة واحدة فقط لأنه لا يعتمد على إعدادات المستخدم
lama_model = LaMaInpainting(model_type='lama', device=device)
print("✅ تم تحميل نموذج إزالة النصوص")

# ذاكرة تخزين مؤقت للنماذج التي تعتمد على إعدادات المستخدم
model_cache = {}

def get_ocr_model(lang):
    """تحميل أو إرجاع نموذج OCR من الذاكرة المؤقتة"""
    if lang not in model_cache:
        print(f"🔄 تحميل نموذج OCR للغة: {lang}")
        model_cache[lang] = PaddleOCR(use_angle_cls=True, lang=lang)
    return model_cache[lang]

def get_realesrgan_model(scale):
    """تحميل أو إرجاع نموذج تحسين الجودة من الذاكرة المؤقتة"""
    key = f"realesrgan_x{scale}"
    if key not in model_cache:
        print(f"🔄 تحميل نموذج تحسين الجودة بمقياس: {scale}x")
        model = RealESRGAN(device, scale=scale)
        # اسم ملف الأوزان يعتمد على المقياس
        model.load_weights(f'RealESRGAN_x{scale}plus.pth', download=True)
        model_cache[key] = model
    return model_cache[key]

# =========================
# دالة معالجة صفحة واحدة
# =========================
def clean_page(image, ocr_model, lama_model, realesrgan_model):
    """تنظيف صفحة واحدة: كشف النص، إزالته، ثم تحسين الجودة"""
    image_np = np.array(image.convert('RGB'))
    h, w = image_np.shape[:2]

    # 1. كشف النصوص
    result = ocr_model.ocr(image_np)
    
    # 2. إنشاء قناع للنصوص
    mask = np.zeros((h, w), dtype=np.uint8)
    if result and result[0]:
        for line in result[0]:
            pts = np.array(line[0], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

    # 3. إزالة النصوص
    cleaned_image = lama_model(image_np, mask)

    # 4. تحسين الجودة
    upscaled_image = realesrgan_model.predict(cleaned_image)

    return Image.fromarray(upscaled_image)

# =========================
# دالة معالجة فصل كامل
# =========================
def clean_chapter_folder(chapter_folder, ocr_model, lama_model, realesrgan_model, output_format):
    """معالجة كل الصور في مجلد فصل واحد"""
    output_folder = os.path.join(chapter_folder, "chapter_cleaned")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # دعم صيغ إضافية
    supported_formats = ('.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp')
    files = sorted([f for f in os.listdir(chapter_folder) if f.lower().endswith(supported_formats)])
    
    if not files:
        print(f"⚠️ لم يتم العثور على صور في المجلد: {chapter_folder}")
        return None

    for idx, file in enumerate(files, 1):
        input_path = os.path.join(chapter_folder, file)
        print(f"📄 معالجة الصفحة {idx}/{len(files)}: {file}")
        
        try:
            image = Image.open(input_path)
            result_image = clean_page(image, ocr_model, lama_model, realesrgan_model)
            
            # حفظ بالصيغة المطلوبة
            file_name, _ = os.path.splitext(file)
            output_filename = f"{file_name}.{output_format.lower()}"
            output_path = os.path.join(output_folder, output_filename)

            if output_format == 'JPG':
                result_image.convert('RGB').save(output_path, 'jpeg', quality=95)
            else:
                result_image.save(output_path, output_format.upper())

        except Exception as e:
            print(f"❌ حدث خطأ أثناء معالجة الملف {file}: {e}")

    return output_folder

# =========================
# الدالة الرئيسية لواجهة Gradio
# =========================
def process_all_chapters(uploaded_files, lang, scale, output_format, progress=gr.Progress(track_tqdm=True)):
    """الدالة التي تربط الواجهة بالمنطق البرمجي"""
    if not uploaded_files:
        return "يرجى اختيار ملفات الصور أولاً."

    # تحميل النماذج بناءً على اختيار المستخدم
    progress(0, desc="تحميل النماذج...")
    ocr_model = get_ocr_model(lang)
    realesrgan_model = get_realesrgan_model(scale)

    # استنتاج المجلد الجذر من قائمة الملفات
    all_paths = [f.name for f in uploaded_files]
    root_path = os.path.commonpath(all_paths) if len(all_paths) > 1 else os.path.dirname(all_paths[0])

    # البحث عن مجلدات الفصول
    potential_chapters = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

    chapters_to_process = []
    if potential_chapters:
        # الحالة 1: تم العثور على مجلدات فرعية (فصول)
        chapters_to_process = [os.path.join(root_path, d) for d in potential_chapters]
        print(f"📂 تم العثور على {len(chapters_to_process)} فصول في: {root_path}")
    else:
        # الحالة 2: لم يتم العثور على مجلدات فرعية، افترض أن المجلد الجذر هو الفصل نفسه
        chapters_to_process = [root_path]
        print(f"📂 لم يتم العثور على فصول، سيتم معالجة المجلد الحالي: {root_path}")

    if not chapters_to_process:
         return "لم يتم العثور على صور للمعالجة في المجلد المحدد."

    processed_chapters_summary = []
    for chapter_path in progress.tqdm(chapters_to_process, desc="معالجة الفصول"):
        chapter_name = os.path.basename(chapter_path)
        print(f"📂 معالجة الفصل: {chapter_name}")
        output_folder = clean_chapter_folder(chapter_path, ocr_model, lama_model, realesrgan_model, output_format)
        if output_folder:
            processed_chapters_summary.append(f"✅ {chapter_name} -> {output_folder}")
        else:
            processed_chapters_summary.append(f"⚠️ {chapter_name} -> لا توجد صور للمعالجة")

    return "\n".join(processed_chapters_summary)

# =========================
# واجهة Gradio
# =========================
interface = gr.Interface(
    fn=process_all_chapters,
    inputs=[
        gr.Files(label="اختر صور الفصول أو اسحب المجلد بأكمله"),
        gr.Dropdown(['en', 'ar', 'ja', 'ko', 'ch_sim', 'fr', 'de'], label="لغة النص في الصور", value='en', info="اختر اللغة لإزالتها بدقة"),
        gr.Dropdown([2, 4], label="مقياس تحسين الجودة", value=2, info="2x هو الأسرع، 4x يعطي جودة أعلى"),
        gr.Radio(['PNG', 'JPG'], label="صيغة حفظ الصور النظيفة", value='PNG')
    ],
    outputs=gr.Textbox(label="الحالة والنتائج", lines=10),
    title="🖌️ منظف المانهوا الذكي المطور",
    description="ارفع الصور التي تريد تنظيفها. يمكنك رفع صور فصل واحد، أو سحب مجلد يحتوي على عدة مجلدات فصول.",
    allow_flagging='never',
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # ملاحظة: عند تشغيل البرنامج لأول مرة، سيتم تحميل وتخزين النماذج. قد يستغرق هذا بعض الوقت.
    # For RealESRGAN scale 4, make sure you have 'RealESRGAN_x4plus.pth' file available or let it download.
    interface.launch()
