# Klasifikasi Sentimen Calon Presiden Dari Pengguna Twitter

Project ini bertujuan untuk melakukan klasifikasi sentimen menggunakan algoritma Naive Bayes terhadap data calon presiden yang diperoleh dari pengguna Twitter. Klasifikasi sentimen ini akan membantu dalam memahami opini publik mengenai calon presiden yang sedang bersaing. Data yang digunakan dalam proyek ini adalah tweet yang mengandung kata-kata kunci terkait calon presiden.

## Permasalahan
### 1. Volume Data yang Besar
  - Data dari Twitter biasanya berjumlah besar, sehingga diperlukan metode yang efisien untuk mengelola dan menganalisis data ini.
### 2. Beragamnya Bahasa dan Slang
  - Tweet sering kali menggunakan bahasa informal, slang, dan singkatan yang dapat menyulitkan proses analisis.
### 3. Noisy Data
  - Data dari media sosial seperti Twitter sering kali mengandung banyak noise, seperti spam, tweet tidak relevan, atau teks yang tidak memiliki makna yang jelas.

## Tujuan
### 1. Mengidentifikasi Sentimen Publik terhadap Setiap Calon Presiden
  - Mengidentifikasi Sentimen Publik terhadap Setiap Calon Presiden.
  - Membandingkan jumlah tweet positif dan negatif untuk setiap calon presiden.
### 2. Mengukur Perubahan Sentimen Seiring Waktu
  - Melacak perubahan sentimen publik terhadap calon presiden dari waktu ke waktu, misalnya, sebelum dan sesudah debat, kampanye, atau peristiwa besar lainnya.
### 3. Menyediakan Data untuk Analisis Strategis Kampanye
  - Menyediakan data yang dapat digunakan oleh tim kampanye untuk mengidentifikasi area yang memerlukan perhatian lebih.
  - Mengidentifikasi sentimen negatif yang mungkin memerlukan tindakan atau klarifikasi dari calon presiden.
### 4. Evaluasi Pengaruh Kampanye Media Sosial
  - Mengevaluasi efektivitas kampanye media sosial masing-masing calon presiden dengan melihat perubahan sentimen setelah kampanye dilakukan.

## Goals
### 1. Pengumpulan Data yang Komprehensif
  - Mengumpulkan data tweet yang relevan dengan kata kunci terkait calon presiden dengan menggunakan API Twitter.
### 2. Pembersihan dan Pra-pemrosesan Data
  - Membersihkan data dari noise, seperti spam dan tweet yang tidak relevan.
  - Normalisasi teks tweet untuk mempersiapkan data bagi analisis lebih lanjut.
### 3. Pengembangan Model Klasifikasi Sentimen yang Akurat
  - Membangun dan melatih model machine learning untuk mengklasifikasikan sentimen tweet sebagai positif, negatif, atau netral dengan akurasi yang tinggi.
### 4. Evaluasi Model dengan Metrik yang Sesuai
  - Menggunakan metrik evaluasi seperti akurasi, presisi, recall, dan F1-score untuk memastikan model memiliki performa yang baik.
### 5. Analisis Sentimen Berdasarkan Calon Presiden
  - Menentukan sentimen publik terhadap masing-masing calon presiden secara keseluruhan dan berdasarkan waktu.
### 6. Pelacakan Perubahan Sentimen Seiring Waktu
  - Melacak dan menganalisis perubahan sentimen publik terhadap calon presiden dari waktu ke waktu, termasuk sebelum dan sesudah peristiwa penting.

## Penjelasan dataset
Saat mencari data untuk menganalisis sentimen pengguna Twitter terhadap calon presiden 2024, langkah pertama yang dilakukan adalah mencari sumber data yang tepat. Salah satu cara untuk mendapatkan data adalah dengan meng-crawl data dari Twitter menggunakan Twitter API. Namun, sebelum memulai proses perayapan, harus mempertimbangkan beberapa pertimbangan, seperti batasan jumlah data yang dikumpulkan dan batas kecepatan API Twitter.
Dalam konteks ini, kami memutuskan untuk menggunakan Twitter API untuk pengumpulan data. Setelah mendaftar sebagai pengembang di platform Twitter dan membangun aplikasi pengembang, kami mulai menyusun kueri penelusuran yang sesuai dengan topik penelitian saya: analisis sentimen calon presiden 2024.
Kueri penelusuran yang kami gunakan mencakup berbagai  nama calon presiden yang relevan dan kata kunci terkait, seperti “Anies”, “Prabowo”, dan “Ganjar”. Setelah membuat permintaan pencarian, saya mulai menerapkan logika dalam program saya untuk memproses data dari Twitter. Pada dataset Anies dan Prabowo memiliki data yang cenderung balance, sedangkan Dataset Ganjar terdapat immbalance data sehingga perlu dilakukan balancing data


## Alur 
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
  - [Confusion Matrix](#confusion-matrix)
- [Hasil dan Kesimpulan](#hasil-dan-kesimpulan)

## Membaca Data

Langkah pertama adalah membaca data yang telah diambil dari Twitter. Disini saya membagi menjadi 3 file data presiden (Anies, Prabowo, Ganjar) dan dimuat ke dalam DataFrame untuk pemrosesan lebih lanjut.

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

Menghapus kata-kata umum yang sering muncul dalam teks tetapi tidak memberikan banyak informasi penting. Kata-kata ini dihapus untuk fokus pada kata-kata yang lebih informatif.

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

### Confusion Matrix

Confusion Matrix digunakan untuk mengevaluasi performa model dengan membandingkan prediksi dengan label yang sebenarnya. Ini membantu dalam mengidentifikasi jenis kesalahan yang dibuat oleh model.

## Hasil dan Kesimpulan
### Hasil
Hasil yang diperoleh dari ketiga capres(Anies, Prabowo, Ganjar) sangat beragam, dan performa model dari ketiga capres juga memiliki hasil yang berbeda, pada capres Ganjar terdapat imbalance data yang harus dilakukan balancing data.
Hasil sentimen dapat dilihat dibawah ini:

### Anies Baswedan
![Anies](https://github.com/user-attachments/assets/064d3aeb-82ca-45ee-858a-d6bd9d3e6da7)
![cmAnies](https://github.com/user-attachments/assets/bfcbcbef-5519-44b5-8a73-8e7e0b38c015)

### Prabowo Subianto
![Prabowo](https://github.com/user-attachments/assets/4154c741-cdce-474f-ae93-ce87fc42c959)
![cmPrabowo](https://github.com/user-attachments/assets/40117759-60a4-4d56-a167-ffd03fb04bdd)

### Ganjar Pranowo (imbalance dataset)
![Ganjar](https://github.com/user-attachments/assets/48f69d80-0245-465d-bc3a-abf8fbb06ca0)
![cmGanjar](https://github.com/user-attachments/assets/90462945-e44a-4c03-a202-21f8afb094c3)

### Kesimpulan 
dapat diambil kesimpulan dari grafik distribusi sentimen dari ketiga capres diatas, Analisis sentimen ini memberikan gambaran umum tentang bagaimana pengguna Twitter merespons ketiga calon presiden. Meskipun sentimen positif dan negatif hadir untuk semua calon, variasi jumlahnya menunjukkan tingkat dukungan dan kritik yang berbeda. Calon Presiden Ganjar tampaknya lebih disukai publik, sementara Calon Presiden Anies menghadapi lebih banyak tantangan dalam mendapatkan dukungan. Calon Presiden Prabowo berada di tengah dengan pendapat publik yang terpolarisasi. Informasi ini dapat digunakan oleh tim kampanye untuk menyusun strategi komunikasi yang lebih efektif dan menangani masalah yang menjadi perhatian publik.
