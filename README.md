# Klasifikasi Sentimen Data Calon Presiden dari Pengguna Twitter

Project ini bertujuan untuk mengklasifikasikan sentimen tweet yang berkaitan dengan presiden menggunakan berbagai teknik preprocessing dan model Gaussian Naive Bayes.

## Daftar Isi
- [Membaca Data](#membaca-data)
- [Pembersihan Data](#pembersihan-data)
  - [Menghapus Duplikat](#menghapus-duplikat)
  - [Menangani Nilai yang Hilang](#menangani-nilai-yang-hilang)
  - [Menghapus Mention, Hashtag, URL, Karakter Khusus, dan Spasi Ekstra](#menghapus-mention-hashtag-url-karakter-khusus-dan-spasi-ekstra)
- [Preprocessing](#preprocessing)
  - [Normalisasi Teks](#normalisasi-teks)
  - [Penghapusan Stopword](#penghapusan-stopword)
  - [Tokenisasi](#tokenisasi)
  - [Stemming](#stemming)
- [Menerjemahkan Data](#menerjemahkan-data)
- [Pelabelan Data](#pelabelan-data)
- [Visualisasi Data](#visualisasi-data)
  - [WordCloud](#wordcloud)
  - [Seaborn](#seaborn)
- [Splitting Data dan Konversi Teks Menjadi Fitur Numerik](#splitting-data-dan-konversi-teks-menjadi-fitur-numerik)
- [Melatih Model dan Membuat Prediksi](#melatih-model-dan-membuat-prediksi)
  - [Confussion Matrix](#confussion-matrix)

## Membaca Data

Langkah pertama adalah membaca data yang telah diambil dari Twitter. Data biasanya disimpan dalam file CSV dan dimuat ke dalam DataFrame untuk pemrosesan lebih lanjut.

## Pembersihan Data

### Menghapus Duplikat

Untuk memastikan setiap tweet adalah unik, entri duplikat dalam dataset dihapus.

### Menangani Nilai yang Hilang

Baris dengan nilai kosong dihapus untuk menjaga kualitas dan integritas data.

### Menghapus Mention, Hashtag, URL, Karakter Khusus, dan Spasi Ekstra

Mentions, hashtag, URL, karakter khusus, dan spasi ekstra dibersihkan dari data teks menggunakan ekspresi reguler. Ini membantu mengurangi kebisingan dalam data dan memastikan hanya informasi relevan yang dipertahankan.

## Preprocessing

### Normalisasi Teks

Teks dinormalisasi dengan mengubah semua huruf menjadi huruf kecil untuk konsistensi dan menghindari variasi yang tidak perlu.

### Penghapusan Stopword

Stopword adalah kata-kata umum yang sering muncul dalam teks tetapi tidak memberikan banyak informasi penting. Kata-kata ini dihapus untuk fokus pada kata-kata yang lebih informatif.

### Tokenisasi

Teks dipecah menjadi unit-unit yang lebih kecil, kata-kata atau token, untuk memudahkan analisis lebih lanjut.

### Stemming

Stemming adalah proses mengubah kata-kata menjadi bentuk dasarnya. Ini membantu mengurangi variasi kata dan meningkatkan konsistensi dalam data.

## Menerjemahkan Data

Data diterjemahkan dari bahasa Indonesia ke bahasa Inggris untuk meningkatkan performa model, khususnya jika model yang digunakan lebih baik dalam bahasa Inggris.

## Pelabelan Data

Data diberi label sesuai dengan sentimen yang terdeteksi, seperti positif, netral, atau negatif, agar model dapat dilatih untuk mengklasifikasikan sentimen dengan benar.

## Visualisasi Data

### WordCloud

WordCloud digunakan untuk memvisualisasikan kata-kata yang paling sering muncul dalam dataset. Ini membantu dalam memahami kata-kata dominan dalam teks.

### Seaborn

Seaborn digunakan untuk visualisasi lain seperti grafik batang atau heatmap untuk menggambarkan distribusi dan hubungan dalam data.

## Splitting Data dan Konversi Teks Menjadi Fitur Numerik

Data dibagi menjadi set pelatihan dan set pengujian. Teks dikonversi menjadi fitur numerik menggunakan teknik seperti `TfidfVectorizer`, yang mengubah teks menjadi representasi numerik yang dapat digunakan oleh model pembelajaran mesin.

## Melatih Model dan Membuat Prediksi

Model Gaussian Naive Bayes dilatih menggunakan data pelatihan. Setelah pelatihan, model digunakan untuk membuat prediksi pada data uji. 

### Confussion Matriks

Matriks kebingungungan digunakan untuk mengevaluasi performa model dengan membandingkan prediksi dengan label yang sebenarnya. Ini membantu dalam mengidentifikasi jenis kesalahan yang dibuat oleh model.
