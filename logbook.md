# Logbook Segmentasi Instance Indoor

## Tujuan dan rancangan eksperimen

Eksperimen membandingkan tiga model instance segmentation indoor: RF-DETR-Seg Nano, YOLO26n-Seg, dan Mask R-CNN ResNet50-FPN. Variabel eksperimen yang tercatat adalah inisialisasi bobot pretrained dan scratch pada tiap arsitektur. Setiap run memakai 50 epoch dan dua GPU Tesla T4 di Kaggle. Parameter lain mengikuti kebutuhan notebook tiap arsitektur. Perbandingan ini tidak menyamakan FLOPs, jumlah parameter, waktu GPU, atau total biaya komputasi.

| Split | Gambar | Anotasi |
|---|---:|---:|
| Train | 397 | 4.479 |
| Valid | 166 | 2.022 |
| Test resmi | 359 | 5.344 |
| **Total** | **922** | **11.845** |

Dataset memiliki 75 kelas. Evaluasi tambahan memakai personal test set terpisah berisi 40 foto nyata, 406 anotasi polygon, dan 42 kelas aktif dari 43 kategori yang dideklarasikan. Anotasi COCO JSON diletakkan di subfolder bernama `train`. Nama itu berasal dari struktur ekspor dan bukan indikasi bahwa foto masuk ke training. Personal test set tetap terpisah dari train dan validasi.

## Hipotesis sebelum training

1. **Model dengan Mask AP tertinggi.** RF-DETR-Seg Nano pretrained diperkirakan memperoleh Mask AP tertinggi secara keseluruhan. Mekanisme attention dan konteks globalnya diperkirakan membantu mengenali objek yang berdekatan atau sebagian tertutup. Bobot pretrained juga diharapkan memberi keunggulan pada dataset yang jumlah data training-nya terbatas.
2. **Pengaruh inisialisasi bobot.** Model pretrained diperkirakan mengungguli model scratch pada ketiga arsitektur. Dataset train hanya memiliki 397 gambar dan training dibatasi 50 epoch, sehingga bobot awal dari pretraining diperkirakan membantu pembelajaran fitur lebih cepat.
3. **Mask pada objek bertumpuk.** Mask R-CNN diperkirakan mampu memisahkan beberapa instance yang saling berdekatan dengan baik karena prediksi mask dibuat untuk tiap region proposal. Keunggulan ini mungkin terlihat pada objek yang bertumpuk, meskipun belum tentu membuat Mask AP keseluruhannya paling tinggi.
4. **Waktu training.** YOLO26n-Seg diperkirakan menyelesaikan training paling cepat karena memakai pendekatan one-stage. Mask R-CNN diperkirakan paling lama karena memiliki tahap proposal region, ekstraksi fitur per region, dan mask head.
5. **Kelas yang lebih mudah dan lebih sulit.** Objek yang relatif besar dan terlihat lebih jelas, seperti bed atau table-merged, diperkirakan lebih mudah disegmentasi daripada objek kecil atau sering tertutup, seperti fork dan spoon. Jumlah instance yang rendah diperkirakan membuat hasil untuk kelas langka lebih sulit.
6. **Perbedaan test resmi dan personal test.** Mask AP pada personal test diperkirakan lebih rendah karena foto diambil sendiri dengan sudut, pencahayaan, dan latar indoor yang mungkin berbeda dari dataset resmi.
7. **Perubahan confidence threshold.** Pada uji postprocessing, confidence threshold yang lebih rendah diperkirakan menaikkan recall sekaligus menambah false positive.

## Alur dan catatan proses

| Tanggal atau tahap | Kegiatan | Kendala dan keputusan |
|---|---|---|
| 19 sampai 24 November | Mengumpulkan dan menganotasi 40 foto personal di Roboflow. | Ekspor anotasi COCO JSON |
| 20 sampai 21 November | Running pretrained dan scratch untuk tiga model, masing-masing 50 epoch di dua Tesla T4. | Split resmi tetap 397 train, 166 valid, dan 359 test. Personal test hanya dipakai untuk evaluasi. |
| Persiapan RF-DETR | Percobaan unduh pretrained checkpoint gagal karena direktori cache belum tersedia. | Direktori cache dibuat sebelum download, lalu training dijalankan kembali. Run final selesai dan log tersimpan. |
| Training Mask R-CNN | Validasi dijalankan setiap epoch, sehingga total waktu run lebih panjang. | `epoch_seconds` mencatat loop training. Total elapsed juga mencakup validasi, jadi keduanya dicatat terpisah. |
| 25 November | Merangkum evaluasi, membandingkan pretrained dan scratch, serta meninjau lima foto kegagalan. | Confidence dan mask IoU diuji pada prediksi tersimpan. Model tidak dilatih ulang untuk eksperimen threshold. |

## Hasil training dan evaluasi

Mask AP ditampilkan sebagai persen. Waktu run mengikuti ringkasan metadata.

| Model | Bobot | Epoch | Mask AP test resmi | Mask AP personal test | Mask AP50 personal test | Waktu run tercatat |
|---|---|---:|---:|---:|---:|---:|
| RF-DETR-Seg Nano | Pretrained | 50 | 14,67% | 15,36% | 21,57% | 26,18 menit |
| RF-DETR-Seg Nano | Scratch | 50 | 0,10% | 0,09% | 0,23% | 25,35 menit |
| YOLO26n-Seg | Pretrained | 50 | 12,36% | 12,24% | 20,67% | 10,59 menit |
| YOLO26n-Seg | Scratch | 50 | 0,01% | 0,01% | 0,08% | 10,11 menit |
| Mask R-CNN | Pretrained | 50 | 7,90% | 11,19% | 17,69% | 268,64 menit* |
| Mask R-CNN | Scratch | 50 | 0,59% | 0,08% | 0,35% | 284,78 menit* |

Pada kedua test set, RF-DETR pretrained mencatat Mask AP tertinggi di antara tiga model pretrained, diikuti YOLO26n-Seg dan Mask R-CNN. Ketiga model pretrained juga mengungguli versi scratch pada personal test. Hasil ini menunjukkan manfaat kuat inisialisasi pretrained pada konfigurasi yang diuji. AP scratch yang rendah belum cukup untuk menyimpulkan overfitting. Kurva training dan validasi diperlukan untuk membedakan model yang belum belajar dari model yang overfit.

Perbedaan hasil test resmi dan personal test belum menjadi ukuran murni domain shift. Test resmi berisi 359 gambar dengan 69 kelas aktif, sedangkan personal test berisi 40 gambar dengan 42 kelas aktif.

### Ciri visual dan domain shift

Statistik visual dihitung per gambar pada thumbnail RGB dengan sisi terpanjang 128 piksel dan rasio aspek dipertahankan. Nilai pada tabel adalah median dari statistik per gambar. Luminansi dihitung dari kanal RGB. Proporsi piksel gelap dan sangat terang memakai ambang luminansi di bawah 50 dan di atas 230 pada skala 0 sampai 255. Ini perbandingan deskriptif, bukan pengukuran kalibrasi kamera.

| Ciri visual | Test resmi, 359 gambar | Personal test, 40 gambar |
|---|---:|---:|
| Rerata kanal RGB per gambar, R / G / B | 149,5 / 137,0 / 126,9 | 130,1 / 121,1 / 113,7 |
| Rerata luminansi | 139,1 | 123,0 |
| Simpangan baku luminansi | 54,7 | 47,9 |
| Piksel gelap, luminansi < 50 | 8,4% | 10,2% |
| Piksel sangat terang, luminansi > 230 | 2,9% | 0,2% |
| Rentang luminansi persentil 95 sampai 5 | 177,3 | 159,9 |
| Rasio lebar terhadap tinggi | 1,35 | 0,56 |

Secara median, foto personal memiliki rerata RGB dan luminansi lebih rendah, kontras luminansi lebih rendah, dan framing lebih portrait. Proporsi piksel sangat terang juga lebih kecil secara global. Ini menunjukkan perbedaan exposure, kontras, dan komposisi antara dua kumpulan foto. Nilai per gambar dipengaruhi objek dan latar, sehingga tidak membuktikan bahwa karakteristik tersebut sendiri menyebabkan perubahan AP. Jumlah gambar dan distribusi kelas kedua test set juga berbeda. Foto backlight tertentu tetap dapat memiliki area terang lokal meskipun median keseluruhan personal test rendah.

## Studi kasus kegagalan

Lima foto dipilih untuk analisis kegagalan RF-DETR pretrained. Pencocokan memakai kelas yang sama, mask IoU minimal 0,50, confidence threshold 0,25, dan pencocokan satu-ke-satu.

| Foto | GT | TP | FP | FN | Temuan |
|---|---:|---:|---:|---:|---|
| `20261124_095547.jpg` | 5 | 3 | 6 | 2 | Foto kamar dengan pencahayaan rendah. Prediksi mencampur bed dan blanket dengan kelas dinding atau meja. |
| `20261124_115953.jpg` | 5 | 2 | 4 | 3 | Objek cardboard berdekatan dengan cup dan meja. Beberapa objek tertukar dengan wall atau wallet. |
| `20261121_130123.jpg` | 7 | 0 | 6 | 7 | Trash bin dan box atau cardboard saling menutupi. Tidak ada prediksi yang lolos pencocokan kelas dan mask. |
| `20261121_140159.jpg` | 14 | 1 | 13 | 13 | Pada tumpukan kardus terjadi kebingungan kelas cardboard dan box. Banyak instance terlewat atau menjadi false positive. |
| `20261124_121049.jpg` | 15 | 3 | 23 | 12 | Peralatan dapur kecil bertumpuk dan saling menutupi. Prediksi tambahan banyak dan sebagian besar instance terlewat. |

Pada lima foto terdapat 46 instance ground truth, 61 prediksi, 9 true positive, 52 false positive, dan 37 false negative. Foto dipilih sebagai kasus kegagalan.

Perbandingan semua model pada lima foto menghasilkan micro F1 0,168 untuk RF-DETR pretrained, 0,169 untuk Mask R-CNN pretrained, 0,102 untuk YOLO26n pretrained, dan mendekati nol untuk model scratch. Pencocokan lintas model memakai resolusi 128 x 128. Hasil tersebut tidak menggantikan peringkat Mask AP pada evaluasi penuh 40 foto dan tidak sebanding langsung dengan tabel kasus RF-DETR yang memakai resolusi asli.

### Interpretasi kegagalan dan kaitan arsitektur

Kasus kardus menunjukkan kebingungan label `cardboard` dan `box`, sedangkan foto kamar dan dapur menunjukkan prediksi latar yang berlebih serta instance kecil atau tertutup yang terlewat. Ini adalah pola yang terlihat pada overlay dan pencocokan class-aware. Di lima foto, YOLO26n pretrained menghasilkan 3 TP dan 43 FN, sedangkan Mask R-CNN pretrained menghasilkan 10 TP dan 62 FP. Pola ini menunjukkan lebih banyak instance terlewat pada YOLO dan banyak prediksi salah pada Mask R-CNN dalam subset terpilih. Karena pencocokan lintas model memakai mask 128 x 128, jumlah itu belum memisahkan kegagalan deteksi dari ketidakcocokan batas mask kecil. Untuk RF-DETR, mekanisme object query membuat persaingan kandidat instance dan penetapan kelas menjadi penjelasan yang masuk akal pada adegan padat, tetapi metrik lima foto tidak membuktikan bahwa query menjadi penyebabnya. YOLO26n-Seg membentuk mask dari prototype dan koefisien, sementara Mask R-CNN memprediksi mask dari fitur tiap region. Representasi tersebut relevan untuk membahas batas mask objek kecil atau saling menutupi. Namun, eksperimen ini tidak mengisolasi resolusi prototype atau resolusi mask per region. Perbandingan lintas model juga mencocokkan mask pada 128 x 128, sehingga tidak cukup untuk menyimpulkan model mana yang menghasilkan batas mask paling akurat.

Uji threshold yang sudah dilakukan menguji apakah sebagian miss dapat dipulihkan dengan melonggarkan confidence. Pada confidence 0,10, TP naik dari 9 menjadi 13 dan FN turun dari 37 menjadi 33, tetapi FP naik dari 52 menjadi 181. Pada confidence 0,50, FP turun menjadi 22, tetapi TP tinggal 5. Jadi threshold memengaruhi trade-off deteksi, tetapi tidak menyelesaikan kebingungan kelas dan tidak menguji langsung pengaruh resolusi mask. Hasil ini cukup untuk menyimpulkan bahwa threshold saja belum mengatasi pola kegagalan yang diamati.

### Uji threshold pada prediksi tersimpan

Uji ini mengubah confidence threshold dan mask IoU pada prediksi tersimpan. Model tidak dilatih ulang.

| Confidence | Mask IoU | Prediksi | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0,10 | 0,50 | 194 | 13 | 181 | 33 |
| 0,25 | 0,50 | 61 | 9 | 52 | 37 |
| 0,50 | 0,50 | 27 | 5 | 22 | 41 |
| 0,25 | 0,25 | 61 | 10 | 51 | 36 |
| 0,25 | 0,75 | 61 | 8 | 53 | 38 |

Menurunkan confidence dari 0,25 ke 0,10 menambah empat TP dan mengurangi empat FN, tetapi menambah 129 FP. Menaikkan confidence ke 0,50 mengurangi 30 FP, tetapi menghilangkan empat TP. Threshold menggeser trade-off precision dan recall. Uji ini belum memperbaiki kebingungan kelas atau kualitas mask.

## Kesimpulan

1. **Hipotesis model terbaik didukung pada evaluasi penuh.** RF-DETR-Seg Nano pretrained mencatat Mask AP tertinggi pada test resmi, 14,67%, dan personal test, 15,36%. Peringkat pretrained pada kedua test adalah RF-DETR, YOLO26n, lalu Mask R-CNN. Pada lima kasus kegagalan terpilih, micro F1 Mask R-CNN sedikit lebih tinggi daripada RF-DETR, 0,169 dibanding 0,168.
2. **Hipotesis pretrained mengungguli scratch didukung kuat.** Pada personal test, Mask AP RF-DETR adalah 15,36% berbanding 0,09%, YOLO26n 12,24% berbanding 0,01%, dan Mask R-CNN 11,19% berbanding 0,08%. Kesimpulan ini berlaku pada konfigurasi 50 epoch yang diuji dan tidak membuktikan bahwa semua kemungkinan konfigurasi scratch akan gagal.
3. **Hipotesis Mask R-CNN lebih baik pada instance bertumpuk belum terbukti.** Pada lima kasus terpilih F1 Mask R-CNN dan RF-DETR hampir sama, sementara Mask R-CNN menghasilkan 62 false positive dan RF-DETR 52. Metrik ini tidak cukup untuk mengklaim kualitas batas mask Mask R-CNN lebih baik.
4. **Hipotesis waktu training didukung pada konfigurasi yang diuji.** YOLO26n mencatat waktu run paling singkat, sekitar 10,11 sampai 10,59 menit. RF-DETR tercatat sekitar 25,35 sampai 26,18 menit. Mask R-CNN tercatat sekitar 268,64 sampai 284,78 menit. Waktu run Mask R-CNN mencakup validasi setiap epoch, sedangkan `epoch_seconds` hanya mencatat loop training, sehingga kedua nilai timer itu tidak dibandingkan langsung.
5. **Hipotesis kelas besar atau langka hanya didukung sebagian.** Pada RF-DETR pretrained, AP bottle 40,70%, book 24,21%, bed 15,82%, chair 10,75%, cardboard 9,65%, box 0%, fork 0%, dan spoon 5,15%. Bottle tampil baik, sedangkan chair tetap rendah meskipun memiliki 89 instance. Jadi ukuran objek dan jumlah contoh saja tidak menjelaskan AP. Kelas dengan sedikit instance juga perlu ditafsirkan hati-hati.
6. **Hipotesis personal test selalu lebih rendah tidak didukung.** Mask AP RF-DETR berubah dari 14,67% pada test resmi menjadi 15,36% pada personal test. YOLO26n berubah dari 12,36% menjadi 12,24%. Mask R-CNN berubah dari 7,90% menjadi 11,19%. Statistik visual menunjukkan personal test memiliki median luminansi dan simpangan baku luminansi lebih rendah, serta framing lebih portrait. Ini mengonfirmasi perbedaan karakteristik visual, tetapi belum membuktikan bahwa perbedaan tersebut menyebabkan perubahan AP. Jumlah gambar dan cakupan kelas juga berbeda.
7. **Hipotesis trade-off threshold didukung pada lima kasus.** Menurunkan confidence dari 0,25 ke 0,10 menambah empat true positive dan mengurangi empat false negative, tetapi menambah 129 false positive. Menaikkan confidence ke 0,50 mengurangi 30 false positive, tetapi kehilangan empat true positive. Threshold mengubah trade-off precision dan recall tanpa memperbaiki kelas yang tertukar.

## Berkas bukti

- Notebook training: `notebooks/rfdetr-seg-nano.ipynb`, `notebooks/yolo26n-seg.ipynb`, dan `notebooks/mask-rcnn.ipynb`
- Log training: `reports/training_logs_for_submission/`
- Metrik inference: `reports/inference_metrics/`
- Ringkasan pretrained dan scratch: `reports/pretrained_vs_scratch.csv` dan `reports/pretrained_gain.csv`
- Audit personal test: `reports/custom_test_audit.json`
- Metadata foto: `reports/custom_test_exif_metadata.csv`
- Statistik visual per gambar: `reports/image_characteristics_per_image.csv`
- Studi lima kasus: `reports/failure_study_v2/`
- Perbandingan lima foto lintas model: `reports/five_photo_cross_model_summary.csv`
- Bukti proses anotasi: `reports/annotation_evidence/`
