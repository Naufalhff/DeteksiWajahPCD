import cv2
import os
import numpy as np

input_folder = "cropped_faces"
output_folder = "datasetaugmentation"
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

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(root, file)
        rel_path = os.path.relpath(root, input_folder)
        output_path = os.path.join(output_folder, rel_path)
        os.makedirs(output_path, exist_ok=True)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Gagal membaca gambar: {img_path}")
            continue

        img = cv2.resize(img, OUTPUT_FACE_SIZE)

        # Simpan ulang gambar asli
        cv2.imwrite(os.path.join(output_path, file), img)

        # Augmentasi
        augmented = {
            "rot15": rotate_image(img, 15),
            "rot-15": rotate_image(img, -15),
            "flip": flip_horizontal(img),
            "bright": adjust_brightness_contrast(img, brightness=30, contrast=0.2),
            "dark": adjust_brightness_contrast(img, brightness=-30, contrast=-0.2),
            "noise": add_gaussian_noise(img),
            "sharp": sharpen_image(img)
        }

        base_filename = os.path.splitext(file)[0]

        for suffix, aug_img in augmented.items():
            aug_name = f"{base_filename}_{suffix}.jpg"
            cv2.imwrite(os.path.join(output_path, aug_name), aug_img)

        print(f"[✓] {img_path} -> augmentasi selesai.")
