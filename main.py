import cv2
import os
import csv
from mtcnn import MTCNN
import numpy as np
from PIL import Image

# Inisialisasi detector
detector = MTCNN()

# Path ke CSV dan folder output
csv_path = "dataset.csv"
output_folder = "cropped_faces"
os.makedirs(output_folder, exist_ok=True)

# Ukuran maksimum gambar input 
MAX_SIZE = 1000
OUTPUT_FACE_SIZE = (512, 512)
MARGIN = 20
CONFIDENCE_THRESHOLD = 0.90

# Ekstensi gambar yang didukung
VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

def load_image(img_path):
    ext = os.path.splitext(img_path)[1].lower()
    img = cv2.imread(img_path)

    if img is None and ext == ".webp":
        try:
            with Image.open(img_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                img = np.array(pil_img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[!] Gagal baca WebP dengan PIL: {img_path} -> {e}")
            return None

    return img

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        img_path = row["path_gambar"]
        nama = row["nama"]
        suku = row["suku"]

        # Cek ekstensi gambar
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in VALID_IMAGE_EXTENSIONS:
            print(f"[!] Format tidak didukung: {img_path}")
            continue

        if not os.path.exists(img_path):
            print(f"[!] File tidak ditemukan: {img_path}")
            continue

        img = load_image(img_path)
        if img is None:
            print(f"[!] Gagal membaca gambar (mungkin corrupt atau format tidak didukung): {img_path}")
            continue

        height, width = img.shape[:2]
        if max(height, width) > MAX_SIZE:
            scale = MAX_SIZE / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            results = detector.detect_faces(img_rgb)
        except Exception as e:
            print(f"[!] Gagal deteksi wajah pada {img_path} -> {e}")
            continue

        if not results:
            print(f"[✓] {img_path} -> 0 wajah ditemukan.")
            continue

        wajah_disimpan = 0
        for i, result in enumerate(results):
            if result["confidence"] < CONFIDENCE_THRESHOLD:
                print(f"[-] Confidence rendah ({result['confidence']:.2f}), dilewati.")
                continue

            x, y, w, h = result['box']
            x, y = max(0, x - MARGIN), max(0, y - MARGIN)
            w, h = w + 2 * MARGIN, h + 2 * MARGIN

            x2, y2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
            face_crop = img[y:y2, x:x2]

            if face_crop.size == 0:
                print(f"[!] Wajah kosong atau crop di luar batas: {img_path}")
                continue

            face_crop_resized = cv2.resize(face_crop, OUTPUT_FACE_SIZE)

            output_path = os.path.join(output_folder, nama, suku)
            os.makedirs(output_path, exist_ok=True)

            filename_base = os.path.splitext(os.path.basename(img_path))[0]
            out_filename = f"{filename_base}.jpg"
            cv2.imwrite(os.path.join(output_path, out_filename), face_crop_resized)
            wajah_disimpan += 1

        print(f"[✓] {img_path} -> {wajah_disimpan} wajah valid disimpan.")
