# Ekstraktor DBF

Aplikasi Streamlit untuk mengekstrak, memfilter, dan mengagregasi data dari satu atau beberapa file DBF
(termasuk file yang dibagi per variabel, mis. Susenas ind1/ind2), dengan penimbang.

## Menjalankan di komputer sendiri
    pip install -r requirements.txt
    streamlit run app.py

## Struktur
- `app.py`               antarmuka Streamlit
- `dbf_core.py`          logika baca-per-chunk, filter, agregasi
- `kamus_variabel.tsv`   kamus kode variabel -> label (dua kolom, dipisah tab)
- `.streamlit/config.toml`  batas unggah file

## Versi online (Streamlit Community Cloud)
Set secret `ALLOW_FOLDER_MODE = "0"` agar mode "baca dari folder" dimatikan.
Jangan pernah menaruh file data (.dbf dll.) di repositori.
