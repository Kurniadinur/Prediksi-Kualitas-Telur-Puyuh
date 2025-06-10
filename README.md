# Klasifikasi Kualitas Telur Burung Puyuh (Quail Egg Quality Classification)

![Aplikasi GUI Klasifikasi Telur Puyuh](https://via.placeholder.com/600x400.png?text=Tangkapan+Layar+Aplikasi+Anda)

Aplikasi ini adalah sistem klasifikasi kualitas telur burung puyuh berbasis GUI (Graphical User Interface) yang dikembangkan menggunakan Python, Tkinter, OpenCV, dan model Machine Learning (Naïve Bayes Classifier serta YOLO untuk deteksi objek). Aplikasi ini mampu menganalisis kualitas telur baik dari gambar statis maupun secara real-time melalui webcam.

## Fitur Utama

* **Klasifikasi Telur**: Mengklasifikasikan kualitas telur burung puyuh menggunakan model Naïve Bayes Classifier.
* **Ekstraksi Fitur Citra**: Mengekstrak fitur-fitur seperti Mean, Variance, dan Range dari channel warna (Red, Green, Blue, Hue, Saturation, Intensity) untuk analisis.
* **Prapemrosesan Gambar**: Menghapus latar belakang objek dan memotong gambar untuk fokus pada telur menggunakan `rembg`.
* **Deteksi Objek Real-time**: Menggunakan model YOLO (You Only Look Once) untuk mendeteksi telur secara real-time dari webcam.
* **Estimasi Jarak**: Memperkirakan jarak telur dari kamera secara real-time.
* **Deteksi Kondisi Pencahayaan**: Menganalisis kondisi pencahayaan (redup, ideal, terang) untuk memastikan kualitas analisis.
* **Antarmuka Pengguna Grafis (GUI)**: Dibangun dengan Tkinter untuk kemudahan penggunaan, dengan tampilan yang telah dimodernisasi menggunakan `ttk` dan layout `grid` serta menampilkan fitur dalam `ttk.Treeview`.

## Instalasi

Untuk menjalankan aplikasi ini, Anda perlu menginstal dependensi Python berikut:

1.  **Kloning repositori ini (jika sudah ada di GitHub):**
    ```bash
    git clone https://github.com/Kurniadinur/Prediksi-Kualitas-Telur-Puyuh.git
    cd YourRepoName
    ```
2.  **Buat Virtual Environment (Sangat Direkomendasikan):**
    ```bash
    python -m venv venv
    ```

3.  **Aktifkan Virtual Environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Instal Dependensi:**
    ```bash
    pip install pandas xlsxwriter Pillow opencv-python numpy rembg ultralytics
    ```
    *Catatan: `tkinter` biasanya sudah terpasang dengan instalasi Python standar. Jika Anda mengalami masalah, Anda mungkin perlu memastikan instalasi Python Anda lengkap atau instalasi terpisah.*

5.  **Siapkan Model Machine Learning:**
    Aplikasi ini memerlukan dua file model:
    * `model 1.pkl`: Model Naïve Bayes Classifier yang telah dilatih.
    * `model/best.pt`: Model YOLO untuk deteksi objek.

    Pastikan `model 1.pkl` berada di *root directory* proyek Anda. Buat folder bernama `model` di *root directory* proyek Anda, dan letakkan file `best.pt` di dalamnya. Struktur folder akan terlihat seperti ini:

    ```
    YourRepoName/
    ├── GUI FIX - Cadangan.py
    ├── model 1.pkl
    └── model/
        └── best.pt
    └── README.md
    ```

## Cara Penggunaan

1.  **Jalankan Aplikasi:**
    Pastikan virtual environment Anda aktif, lalu jalankan script Python:
    ```bash
    python "GUI FIX - Cadangan.C:\Users\Darul\Desktop\GUI FIX - Cadangan.py"
    ```

2.  **Menggunakan Mode Gambar:**
    * Klik tombol **"Buka Gambar"** untuk memilih file gambar telur puyuh dari komputer Anda.
    * Gambar akan ditampilkan di panel "Citra Asli".
    * Klik tombol **"Ekstrak Fitur"** untuk melakukan prapemrosesan gambar (menghapus latar belakang dan memotong) serta mengekstraksi fitur-fitur numerik dari telur.
    * Klik tombol **"Ekstrak"** untuk menampilkan nilai fitur yang diekstraksi di tabel "Ekstraksi Fitur" dan melihat hasil klasifikasi di bawahnya.

3.  **Menggunakan Mode Realtime (Webcam):**
    * Klik tombol **"Realtime"** untuk memulai feed webcam.
    * Aplikasi akan secara otomatis mendeteksi telur, menampilkan kotak pembatas, memperkirakan jarak, dan menampilkan klasifikasi kualitas secara real-time.
    * Tekan tombol `q` atau tutup jendela webcam secara manual untuk menghentikan feed realtime.

## Kontribusi

Jika Anda ingin berkontribusi pada proyek ini, silakan ikuti langkah-langkah berikut:

1.  *Fork* repositori ini.
2.  Buat *branch* baru: `git checkout -b feature/nama-fitur-baru`
3.  Lakukan perubahan Anda dan *commit*: `git commit -m 'Tambahkan fitur baru'`
4.  *Push* ke *branch* Anda: `git push origin feature/nama-fitur-baru`
5.  Buat *Pull Request* baru.
