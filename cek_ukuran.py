
import cv2 as cv
import matplotlib.pyplot as plt

from os import listdir, path
from config import FOLDER_ORI, JUMLAH_KELAS, UKURAN
from Utils.getData import getImageLabel, daftarKelas, EKSTENSI

folder = path.join(FOLDER_ORI.format(k=1), "Train")

# 1. Memeriksa ukuran gambar asli
ukuran_asli = set()

for kelas in daftarKelas(folder):
    for nama in listdir(path.join(folder, kelas)):
        if nama.lower().endswith(EKSTENSI):
            gambar = cv.imread(path.join(folder, kelas, nama))

            if gambar is not None:
                ukuran_asli.add(gambar.shape)

print("=== SEBELUM PREPROCESSING ===")
print("Jumlah variasi ukuran:", len(ukuran_asli))
print("Contoh ukuran:", sorted(ukuran_asli)[:3])

# 2. Memeriksa ukuran setelah preprocessing
data = getImageLabel(folder, JUMLAH_KELAS, UKURAN)

if len(data) == 0:
    raise ValueError("Tidak ada gambar yang berhasil dibaca!")

ukuran_baru = set(
    tuple(gambar.shape) for gambar, label in data
)

print("\n=== SESUDAH PREPROCESSING ===")
print("Ukuran gambar:", ukuran_baru)
print("Jumlah gambar:", len(data))
print("Daftar kelas:", data.kelas)

# 3. Memeriksa normalisasi piksel
gambar, label = data[0]

print("\n=== NORMALISASI ===")
print("Piksel minimum:", gambar.min().item())
print("Piksel maksimum:", gambar.max().item())
print("Contoh label:", label)

# 4. Mengambil satu contoh gambar
folder_mpox = path.join(folder, "Monkeypox")

nama_foto = next(
    n for n in sorted(listdir(folder_mpox))
    if n.lower().endswith(EKSTENSI)
)

asli = cv.imread(path.join(folder_mpox, nama_foto))

if asli is None:
    raise ValueError("Contoh gambar gagal dibaca!")

resize = cv.resize(asli, (UKURAN, UKURAN))

# 5. Visualisasi perbandingan
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(asli, cv.COLOR_BGR2RGB))
plt.title(f"Sebelum: {asli.shape}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(resize, cv.COLOR_BGR2RGB))
plt.title(f"Sesudah: {resize.shape}")
plt.axis("off")

plt.tight_layout()
plt.savefig("cek_ukuran.png", dpi=150)

print("\nGambar berhasil disimpan: cek_ukuran.png")

plt.show()
