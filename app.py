import glob
import io
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import dbf_core as core

st.set_page_config(page_title="Ekstraktor DBF", page_icon="🗂️", layout="wide")
st.title("🗂️ Ekstraktor & Agregator File DBF")
st.caption("Unggah satu atau beberapa file DBF, pilih variabel, tentukan agregasi dan filter, lalu unduh hasilnya.")

# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.header("Pengaturan baca")
    enc_label = st.selectbox(
        "Encoding teks",
        ["auto (dari header DBF)", "cp1252", "cp437", "latin1", "utf-8"],
        help="Ubah jika huruf pada nilai teks tampak rusak.",
    )
    encoding = None if enc_label.startswith("auto") else enc_label
    lowernames = st.checkbox(
        "Ubah nama variabel ke huruf kecil",
        value=True,
        help="Menyamakan nama variabel antar-file yang penulisan huruf besar/kecilnya berbeda.",
    )
    chunk_size = st.number_input(
        "Ukuran chunk (record)", 50_000, 1_000_000, core.CHUNK_SIZE, step=50_000,
        help="Kecilkan jika memori terbatas.",
    )
    kamus = st.file_uploader(
        "Kamus variabel (opsional)", type=["csv", "tsv", "txt"],
        help="Dua kolom: nama variabel dan label (pisah tab atau koma). Jika diunggah, menggantikan "
        "kamus bawaan (file kamus_variabel.tsv di folder aplikasi).",
    )
    default_kamus = Path(__file__).parent / "kamus_variabel.tsv"
    if kamus:
        raw_kamus = kamus.getvalue().decode("utf-8-sig", errors="replace")
    elif default_kamus.exists():
        raw_kamus = default_kamus.read_text(encoding="utf-8-sig")
    else:
        raw_kamus = ""
    labels = core.parse_labels(raw_kamus, lowernames)
    st.caption(f"Kamus aktif: {len(labels)} variabel" if labels else "Kamus variabel belum ada.")

# ------------------------------------------------------------------ Langkah 1
def save_uploads(files) -> list[str]:
    sig = tuple((f.name, f.size) for f in files)
    if st.session_state.get("upload_sig") != sig:
        old = st.session_state.get("tmpdir")
        if old:
            shutil.rmtree(old, ignore_errors=True)
        tmpdir = tempfile.mkdtemp(prefix="dbf_")
        paths = []
        for i, f in enumerate(files):
            p = os.path.join(tmpdir, f"{i:02d}_{os.path.basename(f.name)}")
            with open(p, "wb") as out:
                out.write(f.getbuffer())
            paths.append(p)
        st.session_state.update(upload_sig=sig, tmpdir=tmpdir, upload_paths=paths)
    return st.session_state["upload_paths"]


st.header("1. Unggah file DBF")
# Mode folder membaca file dari komputer/server tempat aplikasi berjalan. Di versi online ini harus dimatikan
# (set variabel lingkungan / secret ALLOW_FOLDER_MODE = "0").
ALLOW_FOLDER = os.environ.get("ALLOW_FOLDER_MODE", "1") != "0"
if ALLOW_FOLDER:
    mode = st.radio(
        "Sumber file",
        ["Unggah dari browser", "Baca dari folder (untuk file sangat besar)"],
        horizontal=True,
    )
else:
    mode = "Unggah dari browser"
    st.caption(
        "File diproses di server aplikasi dan hanya disimpan sementara selama sesi. "
        "File berukuran ratusan MB dapat gagal karena batas memori server; jika itu terjadi, "
        "gunakan file yang lebih kecil atau jalankan aplikasi di komputer sendiri."
    )
if mode.startswith("Unggah"):
    files = st.file_uploader("Pilih satu atau lebih file .dbf", type=["dbf"], accept_multiple_files=True)
    paths = save_uploads(files) if files else []
else:
    folder = st.text_input("Path folder berisi file .dbf", help="Folder harus ada di komputer/server tempat aplikasi berjalan.")
    paths = []
    if folder:
        paths = sorted(
            p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(".dbf")
        )
        if not paths:
            st.warning("Tidak ada file .dbf di folder tersebut.")

if not paths:
    st.info("Menunggu file DBF.")
    st.stop()
paths_t = tuple(paths)


@st.cache_data(show_spinner="Membaca header file...")
def read_schemas(paths_t, encoding, lowernames):
    return {p: core.get_schema(p, encoding, lowernames) for p in paths_t}


schemas = read_schemas(paths_t, encoding, lowernames)
counts = [schemas[p][1] for p in paths]
all_fields = [schemas[p][0] for p in paths]
same_schema = all(set(f) == set(all_fields[0]) for f in all_fields)

st.dataframe(
    pd.DataFrame(
        {
            "file": [os.path.basename(p) for p in paths],
            "jumlah record": counts,
            "jumlah variabel": [len(f) for f in all_fields],
        }
    ),
    hide_index=True,
    width="stretch",
)

# Cara file dibagi: per record (ditumpuk) atau per variabel (digabung menyamping)
data_mode = "stack"
if len(paths) > 1:
    auto_join = (not same_schema) and len(set(counts)) == 1
    mode_label = st.radio(
        "Bagaimana file-file ini dibagi?",
        [
            "Dibagi per record: variabel sama, isi record berbeda (ditumpuk ke bawah)",
            "Dibagi per variabel: variabel berbeda, jumlah record sama (digabung menyamping)",
        ],
        index=1 if auto_join else 0,
        help="Contoh Susenas: ind1.dbf dan ind2.dbf berisi variabel berbeda untuk responden yang sama, "
        "sehingga dipilih 'dibagi per variabel'. Terdeteksi otomatis, tetapi bisa diubah.",
    )
    data_mode = "join" if mode_label.startswith("Dibagi per variabel") else "stack"

first_fields = all_fields[0]
if data_mode == "join":
    if len(set(counts)) != 1:
        st.error("Jumlah record antar-file berbeda, sehingga tidak bisa digabung menyamping. "
                 "Pilih 'dibagi per record' jika file-file ini berisi record yang berbeda.")
        st.stop()
    types = {}
    for f in all_fields:
        for c, t in f.items():
            types.setdefault(c, t)
    common = list(types)
    shared = [c for c in first_fields if all(c in f for f in all_fields)]
    st.success(f"{len(paths)} file digabung menyamping: {counts[0]:,} record, {len(common)} variabel gabungan.")
    if shared:
        st.caption(
            f"Variabel yang ada di semua file ({', '.join(shared[:8])}{'…' if len(shared) > 8 else ''}) "
            "dipakai untuk memeriksa bahwa baris antar-file sejajar. Jika tidak sejajar, proses dihentikan dengan pesan error. "
            "Jika variabel terpilih hanya ada di satu file, file lainnya tidak dibaca (lebih cepat)."
        )
    else:
        st.warning("Tidak ada variabel yang sama antar-file, sehingga kesejajaran baris tidak bisa diperiksa. "
                   "Baris dianggap sejajar berdasarkan urutan.")
else:
    common = [c for c in first_fields if all(c in f for f in all_fields)]
    types = {c: first_fields[c] for c in common}
    st.success(f"{len(paths)} file ditumpuk: total {sum(counts):,} record, {len(common)} variabel yang ada di semua file.")
    if not same_schema:
        st.warning("Struktur variabel antar-file tidak identik. Hanya variabel yang ada di SEMUA file yang dapat dipilih. "
                   "Jika file-file ini sebenarnya dibagi per variabel, pilih opsi 'dibagi per variabel' di atas.")


def fmt(c: str) -> str:
    if c not in types:
        return c
    lab = labels.get(c.lower())
    return f"{c.upper()} — {lab}" if lab else f"{c.upper()} ({types[c]})"


with st.expander("Pratinjau 5 baris pertama"):
    if data_mode == "join":
        for tab, p in zip(st.tabs([os.path.basename(p) for p in paths]), paths):
            with tab:
                st.dataframe(core.preview(p, 5, encoding, lowernames), width="stretch")
    else:
        st.dataframe(core.preview(paths[0], 5, encoding, lowernames), width="stretch")

# ------------------------------------------------------------------ Langkah 2
st.header("2. Pilih variabel yang diekstrak")
selected = st.multiselect(
    "Variabel", common, format_func=fmt,
    help="Termasuk variabel untuk pengelompokan (mis. kode kabupaten/kota) dan variabel yang akan dihitung.",
)

# ------------------------------------------------------------------ Langkah 3
st.header("3. Tentukan variabel penimbang (jika ada)")
w_options = ["(tanpa penimbang)"] + common
default_w = next((c for c in common if c.lower() == "fwt"), None)
w_choice = st.selectbox(
    "Variabel penimbang",
    w_options,
    index=w_options.index(default_w) if default_w else 0,
    format_func=fmt,
    help="Pada data survei (mis. Susenas: FWT) setiap record mewakili sejumlah penduduk. "
    "Pilih '(tanpa penimbang)' untuk menghitung record apa adanya.",
)
weight = None if w_choice == w_options[0] else w_choice
if weight:
    st.info(
        f"Penimbang **{weight}** dipakai. `jumlah_terboboti` = Σ penimbang (estimasi jumlah penduduk); "
        "**sum** = Σ(nilai × penimbang); **mean** = Σ(nilai × penimbang) / Σ penimbang (record bernilai valid). "
        "min, max, dan count tidak dipengaruhi penimbang."
    )
else:
    st.caption("Tanpa penimbang: setiap record bernilai 1.")

# ------------------------------------------------------------------ Langkah 4
st.header("4. Tentukan agregasi")
group_cols = st.multiselect(
    "Agregasi berdasarkan variabel",
    selected,
    format_func=fmt,
    help="Contoh: kode provinsi atau kode kabupaten/kota. Kosongkan untuk total nasional. "
    "Bisa lebih dari satu (mis. provinsi + perkotaan/perdesaan).",
)
if "r102" in [g.lower() for g in group_cols] and "r101" not in [g.lower() for g in group_cols]:
    st.warning("Pada Susenas, kode kabupaten/kota (R102) hanya unik di dalam provinsi. Sertakan juga R101 agar tidak tercampur antar-provinsi.")

rest = [c for c in selected if c not in group_cols and c != weight]
if weight and weight in selected:
    st.caption(f"{weight} dipakai sebagai penimbang, sehingga tidak dihitung sebagai variabel agregasi.")
FN_LABELS = {
    "sum": "sum — jumlah nilai",
    "mean": "mean — rata-rata",
    "min": "min",
    "max": "max",
    "count": "count — jumlah record bernilai angka (tanpa penimbang)",
    "countif": "jumlah (terboboti) bila nilai = …",
    "yatidak": "jumlah YA dan TIDAK (terboboti); Ya bila nilai = …",
    "pctall": "persentase dari SEMUA record bila nilai = …",
    "pctif": "persentase dari record TERISI bila nilai = …",
}
measures: dict[str, str] = {}
targets: dict[str, list[str]] = {}
if rest:
    st.markdown("**Fungsi agregasi untuk variabel yang dihitung**")
    st.caption(
        "Untuk variabel ya/tidak jangan pakai sum. Isi **nilai target** sesuai pengkodean variabel: "
        "`1` jika Ya berkode angka 1; atau `*` jika Ya ditandai sel terisi apa pun dan Tidak dibiarkan kosong "
        "(mis. R811: huruf A, B, C… bila Ya, kosong bila Tidak). Persentase dari SEMUA record cocok untuk "
        "porsi terhadap seluruh penduduk yang difilter; persentase dari record TERISI hanya bila kolom kosong berarti tidak ditanya."
    )

    def _apply_bulk(names):
        for n in names:
            st.session_state[f"fn_{n}"] = st.session_state["bulk_fn"]
            st.session_state[f"tgt_{n}"] = st.session_state["bulk_val"]

    b1, b2, b3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    b1.selectbox("Atur semua variabel sekaligus", core.AGG_FUNCS, index=core.AGG_FUNCS.index("yatidak"), key="bulk_fn", format_func=FN_LABELS.get)
    b2.text_input("Nilai target", value="1", key="bulk_val", help="Untuk fungsi 'bila nilai = …'. Beberapa nilai: 1,2. Tulis * untuk sel terisi apa pun.")
    b3.button("Terapkan ke semua", on_click=_apply_bulk, args=(list(rest),))

    cols = st.columns(min(len(rest), 3))
    for i, c in enumerate(rest):
        with cols[i % len(cols)]:
            measures[c] = st.selectbox(fmt(c), core.AGG_FUNCS, key=f"fn_{c}", format_func=FN_LABELS.get)
            if measures[c] in core.IF_FUNCS:
                st.session_state.setdefault(f"tgt_{c}", "1")
                txt = st.text_input("Nilai target", key=f"tgt_{c}", help="Pisahkan dengan koma jika lebih dari satu. * = sel terisi apa pun (tidak kosong).")
                targets[c] = [v.strip() for v in txt.split(",") if v.strip()]
bad_targets = [c for c, fn in measures.items() if fn in core.IF_FUNCS and not targets.get(c)]
if bad_targets:
    st.warning("Nilai target belum diisi untuk: " + ", ".join(bad_targets))
st.caption("`jumlah_record` selalu disertakan. Nilai non-numerik pada variabel yang dihitung dianggap kosong.")

# ------------------------------------------------------------------ Langkah 4
st.header("5. Tentukan filter")
filter_vars = st.multiselect(
    "Variabel yang difilter", common, format_func=fmt,
    help="Semua filter digabung dengan logika DAN. Variabel filter tidak harus termasuk variabel yang diekstrak.",
)

uniq_all = st.session_state.setdefault("uniq", {})
uniq = uniq_all.setdefault((paths_t, encoding, lowernames, data_mode), {})
missing = [c for c in filter_vars if c not in uniq]
if missing and st.button(f"🔍 Pindai nilai unik: {', '.join(missing)}", help="Membaca seluruh file satu kali."):
    bar = st.progress(0.0, text="Memindai...")
    uniq.update(
        core.scan_columns(paths, missing, encoding, lowernames, lambda f: bar.progress(f, text="Memindai..."), mode=data_mode)
    )
    bar.empty()

filters: list[dict] = []
for c in filter_vars:
    info = uniq.get(c)
    modes = ["Ketik nilai (pisahkan koma)", "Bandingan angka"]
    if info and info["values"] is not None:
        modes.insert(0, "Pilih dari nilai unik")
    with st.container(border=True):
        st.markdown(f"**{fmt(c)}**")
        m = st.radio("Cara filter", modes, horizontal=True, key=f"mode_{c}", label_visibility="collapsed")
        if m == "Pilih dari nilai unik":
            vals = st.multiselect("Nilai yang dipertahankan", info["values"], key=f"val_{c}")
            if vals:
                filters.append({"col": c, "mode": "values", "values": vals})
        elif m.startswith("Ketik"):
            txt = st.text_input("Nilai yang dipertahankan", key=f"txt_{c}", help="Tulis persis seperti di data, mis. 2 atau 3171,3172")
            vals = [v.strip() for v in txt.split(",") if v.strip()]
            if vals:
                filters.append({"col": c, "mode": "values", "values": vals})
        else:
            a, b, d = st.columns(3)
            op = a.selectbox("Operator", ["≥", ">", "≤", "<", "=", "≠", "antara"], key=f"op_{c}")
            default = (info or {}).get("min") or 0.0
            v1 = b.number_input("Nilai", value=float(default), step=1.0, key=f"v1_{c}")
            flt = {"col": c, "mode": "numeric", "op": op, "value": v1}
            if op == "antara":
                v2 = d.number_input("Sampai", value=float((info or {}).get("max") or v1), step=1.0, key=f"v2_{c}")
                flt["value2"] = v2
            filters.append(flt)

# ------------------------------------------------------------------ Langkah 5
st.header("6. Jalankan")
if st.button("▶️ Ekstrak & agregasi", type="primary", disabled=(not selected) or bool(bad_targets)):
    bar = st.progress(0.0, text="Memproses...")
    try:
        result, stats = core.extract(
            paths, group_cols, measures, weight, filters,
            encoding, lowernames, int(chunk_size), lambda f: bar.progress(f, text="Memproses..."),
            mode=data_mode, targets=targets,
        )
    except (ValueError, KeyError) as e:
        bar.empty()
        st.error(str(e))
        st.stop()
    bar.empty()
    params = pd.DataFrame(
        [
            ("File", "; ".join(os.path.basename(p) for p in paths)),
            ("Cara gabung file", "menyamping (per variabel)" if data_mode == "join" else "ditumpuk (per record)"),
            ("Agregasi berdasarkan", ", ".join(group_cols) or "Nasional"),
            ("Penimbang", weight or "-"),
            ("Variabel & fungsi", ", ".join(f"{k}:{v}" + (f"[{','.join(targets[k])}]" if k in targets else "") for k, v in measures.items()) or "-"),
            ("Filter", " DAN ".join(
                f"{f['col']} dalam [{', '.join(map(str, f['values']))}]" if f["mode"] == "values"
                else f"{f['col']} {f['op']} {f['value']}" + (f" s.d. {f['value2']}" if "value2" in f else "")
                for f in filters) or "-"),
            ("Record dibaca", stats["dibaca"]),
            ("Record lolos filter", stats["lolos_filter"]),
        ],
        columns=["Parameter", "Nilai"],
    ).astype(str)
    st.session_state["result"] = (result, stats, params)

if "result" in st.session_state:
    result, stats, params = st.session_state["result"]
    m1, m2 = st.columns(2)
    m1.metric("Record dibaca", f"{stats['dibaca']:,}")
    m2.metric("Record lolos filter", f"{stats['lolos_filter']:,}")
    if result.empty:
        st.warning("Tidak ada record yang lolos filter.")
    else:
        st.dataframe(result, width="stretch", hide_index=True)
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button(
            "⬇️ CSV", result.to_csv(index=False).encode("utf-8-sig"), "hasil_ekstraksi.csv", "text/csv"
        )
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            result.to_excel(xw, sheet_name="Hasil", index=False)
            params.to_excel(xw, sheet_name="Parameter", index=False)
        d2.download_button(
            "⬇️ Excel", buf.getvalue(), "hasil_ekstraksi.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        with st.expander("Parameter yang dipakai"):
            st.dataframe(params, hide_index=True, width="stretch")
