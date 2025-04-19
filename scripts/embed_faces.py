import os
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
from torchvision import transforms

# Path ke folder wajah yang sudah di-crop
BASE_DIR = "cropped_faces"

# Siapkan model FaceNet (InceptionResnetV1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Simpan hasil embedding
embeddings = []

# Transformasi untuk mengubah gambar ke format yang diinginkan oleh model
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Resize ke 160x160
    transforms.ToTensor(),          # Ubah gambar ke tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisasi
])

def get_face_embedding(image_path):
    try:
        # Baca gambar dan ubah ke format RGB
        img = Image.open(image_path).convert('RGB')

        # Transformasi gambar sesuai ukuran input yang diinginkan model
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Ambil embedding dari wajah menggunakan FaceNet
        with torch.no_grad():
            embeddings_face = facenet(img_tensor)

        # Kembalikan hasil embedding sebagai list
        return [val.item() for val in embeddings_face[0]]
    except Exception as e:
        print(f"❌ Gagal proses {image_path}: {e}")
        return None

# Iterasi ke setiap orang di dalam folder
for person in tqdm(os.listdir(BASE_DIR)):
    person_path = os.path.join(BASE_DIR, person)
    if not os.path.isdir(person_path):
        continue

    for ethnicity in os.listdir(person_path):
        ethnicity_path = os.path.join(person_path, ethnicity)
        if not os.path.isdir(ethnicity_path):
            continue

        for image_name in os.listdir(ethnicity_path):
            image_path = os.path.join(ethnicity_path, image_name)

            try:
                # Baca gambar dan ubah ke format RGB
                img = Image.open(image_path).convert('RGB')

                # Transformasi gambar sesuai ukuran input yang diinginkan model
                img_tensor = transform(img).unsqueeze(0).to(device)

                # Ambil embedding dari wajah menggunakan FaceNet
                with torch.no_grad():
                    embeddings_face = facenet(img_tensor)

                # Simpan hasil embedding
                embeddings.append({
                    "path": image_path,
                    "person": person,
                    "ethnicity": ethnicity,
                    **{f"embed_{i}": val.item() for i, val in enumerate(embeddings_face[0])}
                })

            except Exception as e:
                print(f"❌ Gagal proses {image_path}: {e}")

# Simpan ke file CSV
df = pd.DataFrame(embeddings)
df.to_csv("face_embeddings.csv", index=False)
print("✅ Embedding selesai disimpan ke face_embeddings.csv")
