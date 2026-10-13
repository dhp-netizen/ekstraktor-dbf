"""Fungsi inti: membaca DBF per potongan (chunk), memfilter, dan mengagregasi.

Prinsip: data TIDAK pernah dimuat utuh ke memori. Setiap chunk difilter lalu
diagregasi menjadi tabel parsial kecil, kemudian parsial-parsial itu digabung.
Dengan begitu beberapa file DBF besar tetap bisa diproses.
"""
from __future__ import annotations

import csv
import io
import operator
import os
from itertools import islice, zip_longest
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from dbfread import DBF, FieldParser

NUMERIC_TYPES = {"N", "F", "I", "O", "B", "Y"}
MISSING = "(kosong)"
CHUNK_SIZE = 200_000
MAX_UNIQUE = 2000

OPS: dict[str, Callable] = {
    "≥": operator.ge,
    ">": operator.gt,
    "≤": operator.le,
    "<": operator.lt,
    "=": operator.eq,
    "≠": operator.ne,
}
AGG_FUNCS = ["sum", "mean", "min", "max", "count", "countif", "yatidak", "pctall", "pctif"]
IF_FUNCS = {"countif", "yatidak", "pctall", "pctif"}
ANY_FILLED = "*"  # nilai target khusus: sel terisi apa pun (tidak kosong)


# --------------------------------------------------------------------------
# Membaca DBF
# --------------------------------------------------------------------------
class TolerantParser(FieldParser):
    """Nilai numerik yang tidak bisa dibaca (mis. '.', '-', '*') dianggap kosong (None),
    bukan menghentikan proses. Umum pada DBF hasil ekspor SPSS/Stata."""

    def parseN(self, field, data):
        try:
            return super().parseN(field, data)
        except ValueError:
            return None

    def parseF(self, field, data):
        try:
            return super().parseF(field, data)
        except ValueError:
            return None


def open_table(path: str, encoding: str | None = None, lowernames: bool = True) -> DBF:
    return DBF(
        path,
        encoding=encoding,
        lowernames=lowernames,
        load=False,
        ignore_missing_memofile=True,
        char_decode_errors="replace",
        parserclass=TolerantParser,
    )


def get_schema(path: str, encoding=None, lowernames=True) -> tuple[dict[str, str], int]:
    """Kembalikan ({nama_variabel: tipe}, jumlah_record) dari header saja."""
    t = open_table(path, encoding, lowernames)
    return {f.name: f.type for f in t.fields}, int(t.header.numrecords)


def preview(path: str, n: int = 5, encoding=None, lowernames=True) -> pd.DataFrame:
    t = open_table(path, encoding, lowernames)
    return pd.DataFrame([dict(r) for r in islice(iter(t), n)])


def iter_chunks(
    path: str, cols: list[str], encoding=None, lowernames=True, chunk_size=CHUNK_SIZE
) -> Iterator[pd.DataFrame]:
    buf: list[list] = []
    for rec in open_table(path, encoding, lowernames):
        buf.append([rec.get(c) for c in cols])
        if len(buf) >= chunk_size:
            yield pd.DataFrame(buf, columns=cols)
            buf = []
    if buf:
        yield pd.DataFrame(buf, columns=cols)


def parse_labels(raw: str, lowernames: bool = True) -> dict[str, str]:
    """Kamus variabel: dua kolom (Variable, Label), dipisah tab atau koma."""
    sep = "\t" if "\t" in raw else ","
    out: dict[str, str] = {}
    for row in csv.reader(io.StringIO(raw), delimiter=sep, quoting=csv.QUOTE_NONE):
        if len(row) < 2:
            continue
        key = row[0].strip()
        if not key or key.lower() in {"variable", "variable information"}:
            continue
        out[key.lower()] = sep.join(row[1:]).strip()
    return out


CHECK_PRIORITY = ["urut", "r401", "psu", "ssu", "strata"]


def plan_join(paths, cols, encoding=None, lowernames=True):
    """Tentukan variabel mana dibaca dari file mana (file pertama yang memilikinya),
    dan variabel penjaga untuk memeriksa kesejajaran baris antar-file."""
    schemas = [get_schema(p, encoding, lowernames)[0] for p in paths]
    assign: dict[str, list[str]] = {p: [] for p in paths}
    for c in cols:
        for p, s in zip(paths, schemas):
            if c in s:
                assign[p].append(c)
                break
        else:
            raise KeyError(f"Variabel '{c}' tidak ditemukan di file mana pun")
    used = [p for p in paths if assign[p]]
    check: list[str] = []
    if len(used) > 1:
        used_sch = [s for p, s in zip(paths, schemas) if p in used]
        common = [c for c in used_sch[0] if all(c in s for s in used_sch)]
        pri = [c for c in CHECK_PRIORITY if c in common]
        check = (pri + [c for c in common if c not in pri])[:3]
    return assign, used, check


def iter_joined(paths, cols, encoding=None, lowernames=True, chunk_size=CHUNK_SIZE):
    """File dibagi per variabel: baca berdampingan per chunk lalu gabung menyamping.
    Hanya file yang memuat variabel terpilih yang dibaca."""
    assign, used, check = plan_join(paths, cols, encoding, lowernames)
    gens = [
        iter_chunks(p, list(dict.fromkeys(assign[p] + check)), encoding, lowernames, chunk_size)
        for p in used
    ]
    pos = 0
    for parts in zip_longest(*gens):
        if any(x is None for x in parts) or len({len(x) for x in parts}) != 1:
            raise ValueError("Jumlah record antar-file berbeda, file tidak bisa digabung menyamping.")
        base = parts[0]
        for p, df in zip(used[1:], parts[1:]):
            for c in check:
                if not np.array_equal(norm(base[c]).values, norm(df[c]).values):
                    raise ValueError(
                        f"Baris tidak sejajar: nilai '{c}' berbeda antara {os.path.basename(used[0])} dan "
                        f"{os.path.basename(p)} (sekitar record {pos + 1:,}-{pos + len(base):,}). "
                        "Pastikan kedua file berasal dari data yang sama dan urutannya tidak diubah."
                    )
        pos += len(base)
        yield pd.concat([part[assign[p]] for p, part in zip(used, parts)], axis=1)[list(cols)]


def iter_dataset(paths, mode, cols, encoding=None, lowernames=True, chunk_size=CHUNK_SIZE):
    """mode 'stack': file berisi record berbeda (ditumpuk). 'join': file berisi variabel berbeda."""
    if mode == "join":
        yield from iter_joined(paths, cols, encoding, lowernames, chunk_size)
    else:
        for p in paths:
            yield from iter_chunks(p, cols, encoding, lowernames, chunk_size)


def total_records(paths, mode, encoding=None, lowernames=True) -> int:
    counts = [get_schema(p, encoding, lowernames)[1] for p in paths]
    return (counts[0] if mode == "join" else sum(counts)) or 1


# --------------------------------------------------------------------------
# Normalisasi nilai
# --------------------------------------------------------------------------
def norm(s: pd.Series) -> pd.Series:
    """Ubah ke string bersih: strip spasi, buang akhiran '.0', NaN -> (kosong)."""
    out = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return out.where(s.notna(), MISSING)


def norm_values(values) -> set[str]:
    vals = pd.Series(list(values), dtype=object)
    if vals.empty:
        return set()
    return set(norm(vals))


def _sort_key(v: str):
    try:
        return (0, float(v), v)
    except ValueError:
        return (1, 0.0, v)


# --------------------------------------------------------------------------
# Pindai nilai unik (untuk membantu UI filter)
# --------------------------------------------------------------------------
def scan_columns(paths, cols, encoding=None, lowernames=True, progress_cb=None, mode="stack") -> dict:
    uniq = {c: set() for c in cols}
    over = {c: False for c in cols}
    mn = {c: np.inf for c in cols}
    mx = {c: -np.inf for c in cols}
    total = total_records(paths, mode, encoding, lowernames)
    done = 0
    for df in iter_dataset(paths, mode, cols, encoding, lowernames):
        for c in cols:
            if not over[c]:
                uniq[c].update(norm(df[c]).unique())
                if len(uniq[c]) > MAX_UNIQUE:
                    over[c], uniq[c] = True, set()
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().any():
                mn[c], mx[c] = min(mn[c], x.min()), max(mx[c], x.max())
        done += len(df)
        if progress_cb:
            progress_cb(min(done / total, 1.0))
    return {
        c: {
            "values": None if over[c] else sorted(uniq[c], key=_sort_key),
            "min": None if mn[c] == np.inf else float(mn[c]),
            "max": None if mx[c] == -np.inf else float(mx[c]),
        }
        for c in cols
    }


# --------------------------------------------------------------------------
# Filter
# --------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    """filters: [{'col','mode':'values','values':[..]} atau
    {'col','mode':'numeric','op','value'[,'value2']}]. Semua digabung dengan AND."""
    if not filters:
        return df
    mask = np.ones(len(df), dtype=bool)
    for f in filters:
        s = df[f["col"]]
        if f["mode"] == "values":
            m = norm(s).isin(norm_values(f["values"]))
        else:
            x = pd.to_numeric(s, errors="coerce")
            if f["op"] == "antara":
                m = x.between(f["value"], f["value2"])
            else:
                m = OPS[f["op"]](x, f["value"]) & x.notna()
        mask &= np.asarray(m)
    return df[mask]


# --------------------------------------------------------------------------
# Agregasi
# --------------------------------------------------------------------------
def _agg_spec(measures: dict[str, str]) -> dict[str, str]:
    spec = {"_n": "sum", "_w": "sum"}
    for m, fn in measures.items():
        spec.update(
            {
                f"{m}__sum": "sum",
                f"{m}__wv": "sum",
                f"{m}__cnt": "sum",
                f"{m}__min": "min",
                f"{m}__max": "max",
            }
        )
        if fn in IF_FUNCS:
            spec[f"{m}__hit"] = "sum"
            spec[f"{m}__sv"] = "sum"
    return spec


def _partial(df, keys, group_cols, measures, weight, targets) -> pd.DataFrame:
    w = (
        pd.to_numeric(df[weight], errors="coerce").fillna(0.0)
        if weight
        else pd.Series(1.0, index=df.index)
    )
    out = pd.DataFrame(index=df.index)
    if group_cols:
        for k in group_cols:
            out[k] = norm(df[k])
    else:
        out["_all"] = "Nasional"
    out["_n"] = 1
    out["_w"] = w
    for m, fn in measures.items():
        x = pd.to_numeric(df[m], errors="coerce")
        valid = x.notna()
        out[f"{m}__sum"] = (x * w).fillna(0.0)
        out[f"{m}__wv"] = w.where(valid, 0.0)
        out[f"{m}__cnt"] = valid.astype(int)
        out[f"{m}__min"] = x
        out[f"{m}__max"] = x
        if fn in IF_FUNCS:
            s = norm(df[m])
            ok = s.ne(MISSING) & s.ne("")
            tset = norm_values(targets.get(m, []))
            hit = ok if ANY_FILLED in tset else s.isin(tset)
            out[f"{m}__hit"] = w.where(hit & ok, 0.0)
            out[f"{m}__sv"] = w.where(ok, 0.0)
    return out.groupby(keys, as_index=False).agg(_agg_spec(measures))


def _combine(frames, keys, measures) -> pd.DataFrame:
    return (
        pd.concat(frames, ignore_index=True)
        .groupby(keys, as_index=False)
        .agg(_agg_spec(measures))
    )


def _finalize(acc, keys, group_cols, measures: dict[str, str], weight, targets) -> pd.DataFrame:
    res = pd.DataFrame({k: acc[k] for k in keys})
    if not group_cols:
        res = res.rename(columns={"_all": "cakupan"})
    res["jumlah_record"] = acc["_n"].astype(int)
    if weight:
        res[f"jumlah_terboboti ({weight})"] = acc["_w"]
    for m, fn in measures.items():
        if fn == "sum":
            res[f"{m}_sum"] = acc[f"{m}__sum"]
        elif fn == "mean":
            res[f"{m}_mean"] = acc[f"{m}__sum"] / acc[f"{m}__wv"].replace(0, np.nan)
        elif fn == "min":
            res[f"{m}_min"] = acc[f"{m}__min"]
        elif fn == "max":
            res[f"{m}_max"] = acc[f"{m}__max"]
        elif fn == "count":
            res[f"{m}_count"] = acc[f"{m}__cnt"].astype(int)
        elif fn in IF_FUNCS:
            tg = ",".join(sorted(norm_values(targets.get(m, [])), key=_sort_key))
            if fn == "countif":
                res[f"{m}_jumlah[{tg}]"] = acc[f"{m}__hit"]
            elif fn == "yatidak":
                res[f"{m}_ya[{tg}]"] = acc[f"{m}__hit"]
                res[f"{m}_tidak"] = acc["_w"] - acc[f"{m}__hit"]
            elif fn == "pctall":
                res[f"{m}_persen_semua[{tg}]"] = 100.0 * acc[f"{m}__hit"] / acc["_w"].replace(0, np.nan)
            else:
                res[f"{m}_persen_terisi[{tg}]"] = 100.0 * acc[f"{m}__hit"] / acc[f"{m}__sv"].replace(0, np.nan)
    return res.sort_values(list(res.columns[: max(len(group_cols), 1)])).reset_index(drop=True)


def extract(
    paths: list[str],
    group_cols: list[str],
    measures: dict[str, str],
    weight: str | None,
    filters: list[dict],
    encoding: str | None = None,
    lowernames: bool = True,
    chunk_size: int = CHUNK_SIZE,
    progress_cb: Callable[[float], None] | None = None,
    mode: str = "stack",
    targets: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """measures: {variabel: fungsi}, fungsi di AGG_FUNCS.
    targets: {variabel: [nilai, ...]}; nilai '*' = sel terisi apa pun (tidak kosong).
    'countif'  : jumlah (terboboti) record yang nilainya termasuk target.
    'yatidak'  : dua kolom: jumlah Ya (nilai = target) dan jumlah Tidak (= semua record lainnya, termasuk kosong).
    'pctall'   : persen terhadap SELURUH record di kelompok (penyebut = jumlah_terboboti).
    'pctif'    : persen terhadap record yang terisi saja (penyebut = yang tidak kosong).
    Mengembalikan (hasil, statistik)."""
    targets = targets or {}
    for m, fn in measures.items():
        if fn in IF_FUNCS and not targets.get(m):
            raise ValueError(f"Nilai target untuk '{m}' belum diisi (fungsi {fn}).")
    keys = list(group_cols) if group_cols else ["_all"]
    read_cols = list(
        dict.fromkeys(
            list(group_cols)
            + list(measures)
            + ([weight] if weight else [])
            + [f["col"] for f in filters]
        )
    )
    total = total_records(paths, mode, encoding, lowernames)
    n_read = n_pass = 0
    parts: list[pd.DataFrame] = []
    for df in iter_dataset(paths, mode, read_cols, encoding, lowernames, chunk_size):
        n_read += len(df)
        df = apply_filters(df, filters)
        n_pass += len(df)
        if len(df):
            parts.append(_partial(df, keys, group_cols, measures, weight, targets))
            if len(parts) >= 20:
                parts = [_combine(parts, keys, measures)]
        if progress_cb:
            progress_cb(min(n_read / total, 1.0))
    if not parts:
        return pd.DataFrame(), {"dibaca": n_read, "lolos_filter": 0}
    acc = _combine(parts, keys, measures)
    return _finalize(acc, keys, group_cols, measures, weight, targets), {
        "dibaca": n_read,
        "lolos_filter": n_pass,
    }
