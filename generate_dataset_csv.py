import os
import csv

# Path dataset utama
dataset_dir = 'dataset'
output_csv = 'dataset.csv'

rows = []

# Telusuri folder dataset
for nama in os.listdir(dataset_dir):
    path_nama = os.path.join(dataset_dir, nama)
    if os.path.isdir(path_nama):
        for suku in os.listdir(path_nama):
            path_suku = os.path.join(path_nama, suku)
            if os.path.isdir(path_suku):
                for file in os.listdir(path_suku):
                    # Tambahkan dukungan .webp
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        try:
                            # Ekstraksi metadata dari nama file
                            filename_only = os.path.splitext(file)[0]  # buang ekstensi
                            parts = filename_only.split('_')[-1].split(',')  # ambil bagian setelah suku
                            sudut = parts[0] if len(parts) > 0 else ''
                            ekspresi = parts[1] if len(parts) > 1 else ''
                            pencahayaan = parts[2] if len(parts) > 2 else ''
                            
                            # Format path gambar
                            path_gambar = os.path.join(dataset_dir, nama, suku, file).replace("\\", "/")
                            
                            rows.append([path_gambar, nama, suku, ekspresi, sudut, pencahayaan])
                        except Exception as e:
                            print(f"Error processing file {file}: {e}")

# Simpan ke file CSV
with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['path_gambar', 'nama', 'suku', 'ekspresi', 'sudut', 'pencahayaan'])
    writer.writerows(rows)

print(f"✅ CSV berhasil dibuat sebagai '{output_csv}' dengan {len(rows)} data.")
