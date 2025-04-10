import cv2
import os
import csv
from mtcnn import MTCNN

# Inisialisasi detector
detector = MTCNN()

# Path ke CSV dan folder output
csv_path = "dataset.csv"
output_folder = "cropped_faces"
os.makedirs(output_folder, exist_ok=True)

# Ukuran maksimum gambar input agar hemat memori
MAX_SIZE = 1000 
obso
# Ukuran minimum hasil crop wajah (agar tidak blur saat pelatihan/deteksi lanjut)
OUTPUT_FACE_SIZE = (512, 512)

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        img_path = row["path_gambar"]
        nama = row["nama"]
        suku = row["suku"]

        if not os.path.exists(img_path):
            print(f"[!] File tidak ditemukan: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Gagal membaca gambar: {img_path}")
            continue

        # Resize jika gambar terlalu besar
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

        for i, result in enumerate(results):
            x, y, w, h = result['box']
            x, y = max(0, x), max(0, y)

            # Crop dan resize wajah ke ukuran lebih besar agar tidak blur
            face_crop = img[y:y + h, x:x + w]
            if face_crop.size == 0:
                print(f"[!] Wajah kosong atau crop di luar batas: {img_path}")
                continue

            face_crop_resized = cv2.resize(face_crop, OUTPUT_FACE_SIZE)

            # Simpan hasil crop
            output_path = os.path.join(output_folder, nama, suku)
            os.makedirs(output_path, exist_ok=True)

            filename_base = os.path.splitext(os.path.basename(img_path))[0]
            out_filename = f"{filename_base}_face{i}.jpg"
            cv2.imwrite(os.path.join(output_path, out_filename), face_crop_resized)

        print(f"[✓] {img_path} -> {len(results)} wajah ditemukan dan disimpan.")
