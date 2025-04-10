Nama: Naufal Hidayatul Fikri  
NIM: 231511024

---

Deskripsi Proyek

Progres saat ini saya melakukan deteksi wajah otomatis dari gambar menggunakan MTCNN, kemudian meng-crop wajah dan menyimpannya ke folder `cropped_faces/`. Setelah itu, dilakukan proses augmentasi data untuk memperkaya dataset pelatihan agar model lebih tahan terhadap variasi.

Deskripsi Dataset

Dataset terdiri dari total 18 gambar wajah yang berasal dari beberapa suku berbeda. Rinciannya:

-Jawa : 4 gambar
-Sunda : 7 gambar
-Melayu : 2 gambar
-Ambon: 1 gambar
-Batak: 1 gambar
-Lampung: 1 gambar
-Makassar: 1 gambar

Dataset ini disimpan dalam file `dataset.csv` yang berisi path gambar, nama orang, dan asal suku

Fitur Utama

- Deteksi wajah otomatis menggunakan MTCNN
- Crop wajah dan simpan difolder `cropped_faces/`
- Augmentasi data:
  - Rotasi +15° dan -15°
  - Flip horizontal
  - Brightness dan contrast (cerah dan gelap)
  - Gaussian noise ringan
  - Penajaman gambar
- Struktur penyimpanan berdasarkan `nama/suku`
- Augmentasi disimpan dalam folder `datasetaugmentation/`

Cara Menjalankan Program

1. Clone Repository
   git clone https://github.com/Naufalhff/DeteksiWajahPCD.git
   cd DeteksiWajahPCD

2. pip install opencv-python mtcnn

3. jalankan sesuai nama file
   dengan urutan
   py generate_dataset_csv.py
   py main.py
   py augment_faces.py
