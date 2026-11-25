# Deklarasi Penggunaan AI

## Penggunaan AI

ChatGPT digunakan untuk brainstorming, debugging kode, menganalisis error, mendukung dokumentasi, serta merangkum hasil pekerjaan. Data, hasil eksperimen, anotasi, dan isi laporan diperiksa secara manual sebelum pengumpulan. Training dan evaluasi model dijalankan di Kaggle.

## Prompt utama

Prompt utama berikut dirangkum dari instruksi yang digunakan untuk membuat notebook:

> Buat tiga notebook terpisah untuk eksperimen segmentasi instance indoor: RF-DETR-Seg Nano, YOLO26n-Seg, dan Mask R-CNN ResNet50-FPN. Bandingkan pretrained dan scratch selama 50 epoch dengan split resmi 397 train, 166 validasi, dan 359 test. Simpan checkpoint terbaik setiap model agar nanti bisa digunakan untuk evaluasi lokal pada test set berisi 40 foto yang sudah saya anotasi. Simpan juga log training, metrik per epoch dan per kelas, Mask AP, AP50, confusion matrix beserta metode pencocokannya, kurva training, waktu run, informasi GPU, serta output penting lain untuk laporan. Sertakan subsection yang jelas.

## Contoh saran AI yang perlu diperbaiki

Saat menyiapkan training RF-DETR dengan bobot pretrained, perbaikan kode awal belum memastikan folder cache model tersedia. Run gagal dengan `FileNotFoundError` ketika proses download mencoba menyimpan checkpoint ke folder cache yang belum ada.

Perbaikannya adalah membuat folder cache terlebih dahulu, lalu menjalankan proses download checkpoint. Setelah perubahan itu, training berjalan dan log run tersimpan. Contoh ini menunjukkan bahwa saran AI perlu dicocokkan dengan traceback dan diverifikasi di lingkungan training.
