import cv2
import os
import numpy as np

# Folder yang berisi data yang sudah di-split
input_folder = "splits/train"  # Hanya augmentasi data training
output_folder = "splits/train_augmented"
os.makedirs(output_folder, exist_ok=True)

OUTPUT_FACE_SIZE = (512, 512)

# ==== Fungsi Augmentasi ====

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def flip_horizontal(image):
    return cv2.flip(image, 1)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    alpha = 1 + contrast
    beta = brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_gaussian_noise(image, mean=0, sigma=10):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# ==== Proses Augmentasi ====
print("Memulai proses augmentasi data...")
total_augmented = 0

# Salin struktur folder split yang sudah ada
for suku_dir in os.listdir(input_folder):
    suku_path = os.path.join(input_folder, suku_dir)
    if not os.path.isdir(suku_path):
        continue
        
    # Buat folder suku di output
    output_suku_path = os.path.join(output_folder, suku_dir)
    os.makedirs(output_suku_path, exist_ok=True)
    
    # Proses setiap folder nama
    for nama_dir in os.listdir(suku_path):
        nama_path = os.path.join(suku_path, nama_dir)
        if not os.path.isdir(nama_path):
            continue
            
        # Buat folder nama di output
        output_nama_path = os.path.join(output_suku_path, nama_dir)
        os.makedirs(output_nama_path, exist_ok=True)
        
        # Proses setiap gambar
        for file in os.listdir(nama_path):
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(nama_path, file)
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"[!] Gagal membaca gambar: {img_path}")
                continue

            img = cv2.resize(img, OUTPUT_FACE_SIZE)

            # Simpan ulang gambar asli
            original_output = os.path.join(output_nama_path, file)
            cv2.imwrite(original_output, img)

            # Augmentasi
            augmented = {
                "rotasi+15": rotate_image(img, 15),
                "rotasi-15": rotate_image(img, -15),
                "flip": flip_horizontal(img),
                "brightness+": adjust_brightness_contrast(img, brightness=30, contrast=0.2),
                "brightness-": adjust_brightness_contrast(img, brightness=-30, contrast=-0.2),
                "noise": add_gaussian_noise(img),
                "sharp": sharpen_image(img)
            }

            base_filename = os.path.splitext(file)[0]

            for suffix, aug_img in augmented.items():
                aug_name = f"{base_filename}_{suffix}.jpg"
                aug_output = os.path.join(output_nama_path, aug_name)
                cv2.imwrite(aug_output, aug_img)
                total_augmented += 1

            print(f"[✓] {img_path} -> augmentasi selesai.")

print(f"\nProses augmentasi selesai!")
print(f"Total gambar yang diaugmentasi: {total_augmented}")
print(f"Data augmentasi tersimpan di: {output_folder}")

# Membuat metadata baru untuk data yang sudah diaugmentasi
print("\nMenyalin metadata training dan menambahkan entri untuk data augmentasi...")
import pandas as pd

try:
    # Baca metadata training
    train_metadata = pd.read_csv("splits/train_metadata.csv")
    
    # Siapkan DataFrame untuk data augmentasi
    aug_rows = []
    
    # Untuk setiap baris dalam metadata
    for _, row in train_metadata.iterrows():
        # Tambahkan baris asli
        aug_rows.append(row.to_dict())
        
        # Tambahkan baris untuk setiap augmentasi
        base_path = row['path_gambar']
        base_filename = os.path.splitext(os.path.basename(base_path))[0]
        
        for suffix in ["rotasi+15", "rotasi-15", "flip", "brightness+", "brightness-", "noise", "sharp"]:
            aug_row = row.to_dict()
            aug_row['path_gambar'] = f"{os.path.dirname(base_path)}/{base_filename}_{suffix}.jpg"
            aug_rows.append(aug_row)
    
    # Buat DataFrame baru dan simpan
    aug_metadata = pd.DataFrame(aug_rows)
    aug_metadata.to_csv("splits/train_augmented_metadata.csv", index=False)
    print("Metadata untuk data augmentasi telah dibuat: splits/train_augmented_metadata.csv")
    
except Exception as e:
    print(f"Error saat memproses metadata: {e}")