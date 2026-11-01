# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "bitsandbytes>=0.43.0",
#     "datasets",
#     "marimo",
#     "tokenizers>=0.22.0,<=0.23.0",
#     "torch",
#     "torchao>=0.16.0",
#     "torchaudio",
#     "torchcodec",
#     "torchvision",
#     "transformers==5.2.0",
#     "triton>=3.2.0",
#     "trl==0.22.2",
#     "unsloth>=2026.10.5",
#     "unsloth_zoo>=2026.10.5",
#     "uv",
#     "xformers>=0.0.33",
# ]
#
# [tool.uv]
# no-build-package = [
#     "bitsandbytes",
#     "triton",
#     "vllm",
#     "xformers",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import difflib
    import json
    import math
    import random
    import re
    from collections import Counter
    from pathlib import Path

    # ---- HARUS SAMA PERSIS dengan validate_dataset.py dan copilot_prompt.py ----
    SECTION_TAGS = (
        "[1. TRANSKRIP]:",
        "[2. ARTI & KOSA KATA]:",
        "[3. BACA INI (EN)]:",
        "[4. MAKSUD JAWABAN (ID)]:",
    )
    # Template chat Qwen3.5 (non-thinking): giliran assistant TERAKHIR diawali blok think kosong,
    # dan prompt inference selalu diakhiri blok yang sama. Giliran assistant di riwayat TIDAK punya blok ini.
    THINK_BLOCK = "<think>\n\n</think>\n\n"
    INSTRUCTION_PART = "<|im_start|>user\n"
    RESPONSE_PART = "<|im_start|>assistant\n" + THINK_BLOCK

    FORMAT_INSTRUCTION = (
        "Kamu adalah asisten live translator dan wawancara pribadi saya. "
        "Pewawancara berbicara bahasa Inggris. Untuk setiap ucapan pewawancara, "
        "balas SELALU dengan 4 bagian ini, persis urutan dan penandanya:\n"
        "[1. TRANSKRIP]: salin ucapan pewawancara apa adanya\n"
        "[2. ARTI & KOSA KATA]: terjemahan Indonesia + arti istilah penting\n"
        "[3. BACA INI (EN)]: jawaban bahasa Inggris yang natural, orang pertama, siap dibaca keras\n"
        "[4. MAKSUD JAWABAN (ID)]: arti jawaban tadi dalam bahasa Indonesia\n"
        "Panjang bagian 3: obrolan ringan 1-3 kalimat; pertanyaan substantif "
        "(pengalaman, teknis, situasional) sekitar 90-130 kata dengan alasan, trade-off, "
        "dan hasil konkret dari profil.\n"
        "Sumber: fakta pribadi HANYA dari PROFIL; konsep teknis umum utamakan KARTU PENGETAHUAN "
        "(kalau tidak ada di kartu, jawab hanya bila yakin, ringkas dan benar); "
        "info perusahaan HANYA dari LOWONGAN (kalau kosong, jawab generik dan jujur). "
        "Kalau sesuatu tidak ada di profil, jujur bilang belum pernah, lalu kaitkan dengan pengalaman "
        "terdekat yang ada. Kalau pertanyaan tidak jelas atau di luar kemampuan, beri kalimat penunda "
        "sopan dan pertanyaan klarifikasi. Jangan mengarang angka, perusahaan, atau teknologi."
    )
    FALLBACK_SYSTEM = "Kamu adalah asisten live translator dan wawancara pribadi saya."

    SUBSTANTIVE = {"behavioral", "tech_concept", "tech_practice", "situational"}
    STOPWORDS = {
        "the", "and", "to", "of", "in", "is", "it", "that", "for", "with", "on", "was", "we", "my", "our", "i",
        "a", "an", "at", "as", "by", "from", "have", "had", "this", "you", "your", "are", "so", "can", "how",
        "about", "not", "but", "or", "be", "been", "into", "than", "then", "when", "what", "which", "they",
        "their", "would", "could", "also", "more", "most", "some", "such", "very", "just", "were", "will",
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu", "pada", "adalah", "saya", "kami",
        "kamu", "anda", "tidak", "juga", "bisa", "ada", "karena", "sebagai", "dalam", "atau", "akan", "sudah",
        "belum", "lebih", "saat", "jadi", "oleh", "agar", "secara", "apa", "bagaimana", "kenapa",
    }


@app.function
def build_system_prompt(profile_text=None, card_text=None, jd_text=None):
    """System prompt = instruksi + PROFIL + KARTU PENGETAHUAN + LOWONGAN. Tanpa profil -> kalimat tunggal (mode hafalan)."""
    profile = (profile_text or "").strip()
    if not profile:
        return FALLBACK_SYSTEM
    parts = [FORMAT_INSTRUCTION, "PROFIL:\n" + profile]
    card = (card_text or "").strip()
    if card:
        parts.append("KARTU PENGETAHUAN:\n" + card)
    jd = (jd_text or "").strip()
    parts.append("LOWONGAN:\n" + (jd if jd else "(kosong)"))
    return "\n\n".join(parts)


@app.function
def parse_jd_bank(text):
    """Format: baris '=== jd1 ===' lalu isi lowongan, diulang untuk jd2, jd3, ..."""
    banks, cur, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^===\s*([A-Za-z0-9_\-]+)\s*===\s*$", line.strip())
        if m:
            if cur:
                banks[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1), []
        elif cur is not None:
            buf.append(line)
    if cur:
        banks[cur] = "\n".join(buf).strip()
    return banks


@app.function
def row_jd(row, jd_bank):
    if row.get("jd"):
        return str(row["jd"])
    jid = row.get("jd_id")
    if jid:
        if jid not in jd_bank:
            raise ValueError(f"jd_id '{jid}' tidak ada di bank lowongan")
        return jd_bank[jid]
    return None


@app.function
def read_jsonl_text(text):
    rows = []
    for ln, raw in enumerate(text.splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"baris {ln}: JSON tidak valid ({e.msg}). Jalankan validate_dataset.py dulu.") from e
        if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
            raise ValueError(f'baris {ln}: harus objek dengan kunci "messages".')
        rows.append(row)
    return rows


@app.function
def conversation_messages(row, profile_text, card_text, jd_bank):
    """Satu baris JSONL -> [system, user, assistant, ...]. System disusun per baris (profil + kartu + lowongan baris itu)."""
    jd = row_jd(row, jd_bank) if (profile_text or "").strip() else None  # mode hafalan mengabaikan lowongan
    system_prompt = build_system_prompt(profile_text, card_text, jd)
    body = [
        {"role": m["role"], "content": m["content"]}
        for m in row["messages"]
        if m.get("role") != "system"
    ]
    return [{"role": "system", "content": system_prompt}] + body


@app.function
def split_conversations(rows, val_ratio, seed):
    """Split per PERCAKAPAN (bukan per giliran) supaya tidak bocor ke validasi. Stratified per category."""
    rng = random.Random(seed)
    groups = {}
    for r in rows:
        groups.setdefault(r.get("category", "_"), []).append(r)
    train, val = [], []
    for cat in sorted(groups):
        g = groups[cat][:]
        rng.shuffle(g)
        n_val = round(len(g) * val_ratio)
        if len(g) >= 8:
            n_val = max(1, n_val)
        val += g[:n_val]
        train += g[n_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


@app.function
def expand_turns(messages):
    """Percakapan multi-turn -> satu contoh per giliran assistant, masing-masing berakhir di assistant itu.
    Dengan begitu SETIAP jawaban yang dilatih dirender persis seperti saat inference (blok think kosong)."""
    return [messages[: i + 1] for i, m in enumerate(messages) if m["role"] == "assistant"]


@app.function
def render_train_text(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    # Buang "\n" penutup: SFTTrainer menambah EOS sendiri bila teks tidak berakhir di <|im_end|>.
    return text.removesuffix("\n")


@app.function
def render_prompt(tokenizer, messages):
    """Prompt inference PERSIS seperti yang harus diterima model saat live."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


@app.function
def check_rendered(text):
    problems = []
    if text.count(RESPONSE_PART) != 1:
        problems.append(f"penanda respons muncul {text.count(RESPONSE_PART)}x (harus 1x)")
    if not text.endswith("<|im_end|>"):
        problems.append("tidak berakhir di <|im_end|>")
    return problems


@app.function
def build_texts(tokenizer, rows, profile_text, card_text, jd_bank):
    texts = []
    for n, row in enumerate(rows, 1):
        for part in expand_turns(conversation_messages(row, profile_text, card_text, jd_bank)):
            text = render_train_text(tokenizer, part)
            bad = check_rendered(text)
            if bad:
                raise ValueError(f"percakapan ke-{n}: render tidak sesuai kontrak: {bad}")
            texts.append(text)
    return texts


@app.function
def parse_sections(text):
    """(dict, None) kalau 4 bagian valid, selain itu (None, alasan)."""
    tags = SECTION_TAGS
    if not text.startswith(tags[0]):
        return None, f"tidak diawali {tags[0]}"
    positions = []
    for tag in tags:
        count = text.count(tag)
        if count != 1:
            return None, f"penanda {tag} muncul {count}x"
        positions.append(text.index(tag))
    if positions != sorted(positions):
        return None, "urutan bagian salah"
    out = {}
    for i, tag in enumerate(tags):
        start = positions[i] + len(tag)
        end = positions[i + 1] if i + 1 < len(tags) else len(text)
        body = text[start:end].strip()
        if not body:
            return None, f"isi {tag} kosong"
        out[tag] = body
    return out, None


@app.function
def trim_history(messages, max_pairs=4):
    """Untuk live: messages = [user, assistant, ..., user(baru)] TANPA system.
    Simpan max_pairs giliran sebelumnya + pertanyaan baru, dan pastikan diawali 'user'."""
    kept = messages[-(2 * max_pairs + 1):]
    while kept and kept[0]["role"] != "user":
        kept = kept[1:]
    return kept


@app.function
def vocab_from(*texts):
    """Kosakata 'detail' dari profil/kartu/lowongan: kata >= 4 huruf (bukan stopword) dan angka >= 2 digit."""
    vocab = set()
    for t in texts:
        for w in re.findall(r"[a-z0-9]+", (t or "").lower()):
            if (w.isdigit() and len(w) >= 2) or (len(w) >= 4 and w not in STOPWORDS):
                vocab.add(w)
    return vocab


@app.function
def overlap_count(text, vocab):
    return len({w for w in re.findall(r"[a-z0-9]+", text.lower()) if w in vocab})


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fine-tune Qwen3.5 untuk Asisten Wawancara (Unsloth di Molab)

    **Urutan kerja**
    1. Klik *notebook specs* di header Molab → **aktifkan GPU**.
    2. Drop SEMUA file sekaligus ke kotak upload: `dataset_cv.jsonl`, `cv_profile.txt`, `knowledge_card.txt`, `lowongan_bank.txt` (nama boleh berakhiran `.example`/`.demo`). Validasi dulu di laptop dengan `validate_dataset.py`.
    3. Pilih model di cell konfigurasi: `unsloth/Qwen3.5-2B` (muat di laptop 4GB) atau `unsloth/Qwen3.5-4B` (cek dulu apakah muat di GPU-mu). Untuk ganti model: ubah `MODEL_NAME`, **restart kernel** (VRAM bersih), Run all. Folder keluaran otomatis dibedakan per model.
    4. **Run all**. Training dan evaluasi baru jalan setelah kamu klik tombolnya.
    5. Pantau **eval loss per epoch**; kalau naik di epoch terakhir, kurangi `EPOCHS`.
    6. Simpan LoRA (kecil, cepat) lalu export GGUF.

    **Catatan penting**
    - System prompt disusun **per baris**: PROFIL + KARTU PENGETAHUAN + LOWONGAN (dari `jd_id`/`jd` baris itu, atau `(kosong)`). Saat live, pakai `copilot_prompt.py` supaya susunannya sama.
    - bf16 LoRA, bukan QLoRA (Unsloth tidak menyarankan QLoRA untuk Qwen3.5). Menurut Unsloth, LoRA bf16 butuh sekitar 5 GB VRAM untuk 2B dan 10 GB untuk 4B (pada panjang konteks standar); GPU Molab 96 GB lebih dari cukup.
    - Loss hanya dihitung di **jawaban asisten terakhir** tiap contoh (`train_on_responses_only`), bukan di ucapan pewawancara, system prompt, atau riwayat.
    - Training tidak jalan sendiri; harus klik tombolnya. Setelah drop file, cell di bawahnya otomatis jalan (data, trainer, cek masking). Mengubah konfigurasi (mis. `MODEL_NAME`) otomatis memuat ulang model; untuk ganti model, restart kernel dulu supaya VRAM bersih.
    - Kalau OOM: turunkan `PER_DEVICE_BATCH` ke 2 dan naikkan `GRAD_ACCUM` ke 8.
    """)
    return


@app.cell
def _():
    import subprocess

    # Resep resmi Unsloth untuk Molab: pasang kernel khusus Qwen3.5.
    try:
        _r = subprocess.run(["unsloth", "install-kernels"], check=False)
        if _r.returncode != 0:
            print(
                f"`unsloth install-kernels` berakhir dengan kode {_r.returncode}: training Qwen3.5 bisa lebih lambat. "
                "Lanjutkan, tapi sertakan pesan ini kalau melapor error."
            )
    except FileNotFoundError:
        print("Perintah `unsloth` belum ada di PATH. Pastikan dependensi di header notebook ter-install.")
    return


@app.cell
def _(mo):
    import torch

    if torch.cuda.is_available():
        _p = torch.cuda.get_device_properties(0)
        _msg = f"✅ GPU terdeteksi: **{_p.name}** ({_p.total_memory / 1e9:.0f} GB VRAM)"
    else:
        _msg = "⚠️ GPU **tidak** terdeteksi. Aktifkan GPU lewat *notebook specs* di header Molab, lalu restart notebook."
    mo.md(_msg)
    return


@app.cell
def _(mo):
    from importlib.metadata import PackageNotFoundError, version

    _v = []
    for _p in ["torch", "transformers", "trl", "unsloth", "unsloth_zoo", "datasets", "peft"]:
        try:
            _v.append(f"{_p} {version(_p)}")
        except PackageNotFoundError:
            _v.append(f"{_p} (tidak terpasang)")
    mo.md("**Versi paket:** " + " | ".join(_v) + "\n\n_Sertakan baris ini kalau melapor error._")
    return


@app.cell
def _():
    # ---------------- KONFIGURASI ----------------
    MODEL_NAME = "unsloth/Qwen3.5-2B"  # atau "unsloth/Qwen3.5-4B"
    RUN_TAG = MODEL_NAME.split("/")[-1].lower()
    MAX_SEQ_LENGTH = 4096  # system prompt kaya ~1.6-1.8K token + jawaban; jangan lebih kecil dari max di validator
    LORA_R = 16
    EPOCHS = 3  # Unsloth: 1-3 epoch; >3 biasanya overfit
    LEARNING_RATE = 2e-4
    PER_DEVICE_BATCH = 4
    GRAD_ACCUM = 4  # batch efektif 16 (rekomendasi Unsloth)
    VAL_RATIO = 0.10
    SEED = 3407
    EVAL_N = 20  # jumlah contoh validasi yang digenerate untuk cek format & kedalaman

    DATASET_FILE = "dataset_cv.jsonl"
    PROFILE_FILE = "cv_profile.txt"
    CARD_FILE = "knowledge_card.txt"
    JD_BANK_FILE = "lowongan_bank.txt"
    OUTPUT_DIR = f"outputs_cv_{RUN_TAG}"
    LORA_DIR = f"asisten_cv_lora_{RUN_TAG}"
    GGUF_DIR = f"asisten_cv_gguf_{RUN_TAG}"
    return (
        CARD_FILE,
        DATASET_FILE,
        EPOCHS,
        EVAL_N,
        GGUF_DIR,
        GRAD_ACCUM,
        JD_BANK_FILE,
        LEARNING_RATE,
        LORA_DIR,
        LORA_R,
        MAX_SEQ_LENGTH,
        MODEL_NAME,
        OUTPUT_DIR,
        PER_DEVICE_BATCH,
        PROFILE_FILE,
        SEED,
        VAL_RATIO,
    )


@app.cell
def _(mo):
    uploads = mo.ui.file(
        kind="area",
        multiple=True,
        filetypes=[".jsonl", ".txt", ".md"],
        label="Drop semua file sekaligus (kalau belum ada di folder notebook): dataset .jsonl, cv_profile, knowledge_card, lowongan_bank",
    )
    uploads
    return (uploads,)


@app.cell
def _(CARD_FILE, DATASET_FILE, JD_BANK_FILE, PROFILE_FILE, mo, uploads):
    _up = {}
    for _i in range(len(uploads.value)):
        _name = (uploads.name(_i) or "").lower()
        _text = uploads.contents(_i).decode("utf-8")
        if _name.endswith(".jsonl"):
            _up["data"] = (uploads.name(_i), _text)
        elif "profile" in _name:
            _up["profile"] = (uploads.name(_i), _text)
        elif "knowledge" in _name or "kartu" in _name:
            _up["card"] = (uploads.name(_i), _text)
        elif "lowongan" in _name or "jd_bank" in _name:
            _up["bank"] = (uploads.name(_i), _text)

    def _from_disk(path):
        p = Path(path)
        return (path, p.read_text(encoding="utf-8")) if p.exists() else None

    _data = _up.get("data") or _from_disk(DATASET_FILE)
    mo.stop(
        _data is None,
        mo.md(f"⚠️ Belum ada dataset. Drop file `.jsonl` di atas, atau taruh `{DATASET_FILE}` di folder notebook."),
    )
    _prof = _up.get("profile") or _from_disk(PROFILE_FILE)
    _card = _up.get("card") or _from_disk(CARD_FILE)
    _bank = _up.get("bank") or _from_disk(JD_BANK_FILE)

    profile_text = _prof[1] if _prof else ""
    card_text = _card[1] if _card else ""
    jd_bank = parse_jd_bank(_bank[1]) if _bank else {}
    rows = read_jsonl_text(_data[1])

    _missing = (
        sorted({r["jd_id"] for r in rows if r.get("jd_id") and not r.get("jd") and r["jd_id"] not in jd_bank})
        if profile_text.strip()
        else []
    )
    mo.stop(
        bool(_missing),
        mo.md(f"⚠️ `jd_id` berikut tidak ada di bank lowongan: {_missing}. Drop `lowongan_bank.txt` di atas."),
    )

    _cats = {}
    for _r in rows:
        _k = _r.get("category", "(tanpa kategori)")
        _cats[_k] = _cats.get(_k, 0) + 1
    _n_turns = sum(sum(m.get("role") == "assistant" for m in r["messages"]) for r in rows)
    _with_jd = sum(1 for r in rows if r.get("jd") or r.get("jd_id"))
    _notes = []
    if not profile_text.strip():
        _notes.append("⚠️ **Tanpa profil → mode HAFALAN**: system prompt satu kalimat, fakta CV harus dihafal bobot model (kurang andal untuk model kecil). Kartu dan lowongan diabaikan.")
    elif not card_text.strip():
        _notes.append("ℹ️ Tanpa kartu pengetahuan: konsep teknis umum hanya mengandalkan pengetahuan bawaan model.")
    mo.md(
        f"✅ **{len(rows)}** percakapan → **{_n_turns}** contoh training setelah expand per giliran. "
        f"Memakai lowongan: **{_with_jd}** baris. Kategori: {_cats}.\n\n"
        f"Profil: {len(profile_text)} karakter | Kartu: {len(card_text)} karakter | Bank lowongan: {len(jd_bank)} entri."
        + ("\n\n" + "\n\n".join(_notes) if _notes else "")
    )
    return card_text, jd_bank, profile_text, rows


@app.cell
def _():
    # PENTING: import unsloth PERTAMA (sebelum trl / transformers / peft).
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    return (
        Dataset,
        FastLanguageModel,
        SFTConfig,
        SFTTrainer,
        is_bfloat16_supported,
        train_on_responses_only,
    )


@app.cell
def _(FastLanguageModel, LORA_R, MAX_SEQ_LENGTH, MODEL_NAME, SEED):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,  # QLoRA tidak disarankan untuk Qwen3.5
        load_in_16bit=True,  # bf16 / 16-bit LoRA
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention penuh (6 dari 24 layer pada 2B)
            "in_proj_qkv", "in_proj_z", "out_proj",  # linear-attention / Gated DeltaNet (18 dari 24 layer)
            "gate_proj", "up_proj", "down_proj",  # MLP (semua layer)
        ],
        lora_alpha=LORA_R,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    # Qwen3.5 adalah VLM: `tokenizer` bisa berupa Processor. Untuk render teks pakai tokenizer di dalamnya.
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    return model, text_tok, tokenizer


@app.cell
def _(mo, model):
    # Diagnostik: adapter LoRA benar-benar terpasang di mana?
    _c = Counter()
    for _n, _m in model.named_modules():
        if hasattr(_m, "lora_A") and "visual" not in _n:
            _c[_n.split(".")[-1]] += 1
    _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in model.parameters())
    mo.md(
        f"**Adapter LoRA terpasang (jumlah modul per nama):** {dict(sorted(_c.items()))}\n\n"
        f"**Parameter terlatih:** {_trainable / 1e6:.1f}M dari {_total / 1e6:.0f}M ({_trainable / _total:.2%})\n\n"
        "_Harapan untuk Qwen3.5-2B (24 layer = 6 attention penuh + 18 linear-attention): "
        "q/k/v/o_proj = 6, in_proj_qkv/in_proj_z/out_proj = 18, gate/up/down_proj = 24. "
        "Kalau angkanya beda jauh, sertakan baris ini saat melapor._"
    )
    return


@app.cell
def _(
    Dataset,
    MAX_SEQ_LENGTH,
    SEED,
    VAL_RATIO,
    card_text,
    jd_bank,
    mo,
    profile_text,
    rows,
    text_tok,
):
    train_rows, val_rows = split_conversations(rows, VAL_RATIO, SEED)
    train_texts = build_texts(text_tok, train_rows, profile_text, card_text, jd_bank)
    val_texts = build_texts(text_tok, val_rows, profile_text, card_text, jd_bank)

    _lens = sorted(len(text_tok(t, add_special_tokens=False)["input_ids"]) for t in train_texts + val_texts)
    _max = _lens[-1]
    if _max > MAX_SEQ_LENGTH:
        raise ValueError(
            f"Ada contoh {_max} token > MAX_SEQ_LENGTH {MAX_SEQ_LENGTH}. Terpotong = penanda respons hilang. "
            "Ringkas profil/kartu/lowongan atau naikkan MAX_SEQ_LENGTH."
        )
    _sys_tokens = len(
        text_tok(build_system_prompt(profile_text, card_text, None), add_special_tokens=False)["input_ids"]
    )

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts}) if val_texts else None

    mo.md(
        f"**Train:** {len(train_rows)} percakapan → {len(train_texts)} contoh | "
        f"**Val:** {len(val_rows)} percakapan → {len(val_texts)} contoh\n\n"
        f"**System prompt dasar:** {_sys_tokens} token | **Panjang contoh:** p50={_lens[len(_lens) // 2]}, max={_max} (batas {MAX_SEQ_LENGTH})\n\n"
        "**Contoh hasil render (ekor, ini yang dilihat model saat training):**\n"
        f"```\n{train_texts[0][-650:]}\n```"
    )
    return train_ds, val_ds, val_rows


@app.cell
def _(
    EPOCHS,
    GRAD_ACCUM,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    OUTPUT_DIR,
    PER_DEVICE_BATCH,
    SEED,
    SFTConfig,
    SFTTrainer,
    is_bfloat16_supported,
    mo,
    model,
    tokenizer,
    train_ds,
    train_on_responses_only,
    val_ds,
):
    _steps = math.ceil(len(train_ds) / (PER_DEVICE_BATCH * GRAD_ACCUM)) * EPOCHS
    _warmup = max(3, round(0.1 * _steps))  # Unsloth: 5-10% dari total step

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=_warmup,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="linear",
            weight_decay=0.01,
            optim="adamw_8bit",
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            logging_steps=5,
            eval_strategy="epoch" if val_ds is not None else "no",
            save_strategy="no",
            seed=SEED,
            report_to="none",
            output_dir=OUTPUT_DIR,
            padding_free=False,
            dataset_num_proc=1,
        ),
    )
    # Loss hanya di jawaban asisten TERAKHIR tiap contoh (system, user, dan riwayat di-mask).
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
        last_response_only=True,
    )
    mo.md(f"Trainer siap: ~{_steps} step total, warmup {_warmup} step, batch efektif {PER_DEVICE_BATCH * GRAD_ACCUM}.")
    return (trainer,)


@app.cell
def _(mo, text_tok, trainer):
    # CEK MASKING sebelum membuang waktu GPU: yang dilatih harus = jawaban asisten, dari "[1. TRANSKRIP]:" sampai <|im_end|>.
    _bad, _fracs = [], []
    _n = min(len(trainer.train_dataset), 200)
    for _i in range(_n):
        _lab = trainer.train_dataset[_i]["labels"]
        _lab = _lab.tolist() if hasattr(_lab, "tolist") else list(_lab)
        _kept = [t for t in _lab if t != -100]
        _fracs.append(len(_kept) / max(len(_lab), 1))
        _txt = text_tok.decode(_kept)
        if not (_txt.startswith(SECTION_TAGS[0]) and _txt.rstrip().endswith("<|im_end|>")):
            _bad.append(_i)
    _lab0 = trainer.train_dataset[0]["labels"]
    _lab0 = _lab0.tolist() if hasattr(_lab0, "tolist") else list(_lab0)
    _sample = text_tok.decode([t for t in _lab0 if t != -100])
    if _bad:
        raise ValueError(
            f"Masking salah pada {len(_bad)} dari {_n} contoh (indeks pertama: {_bad[:5]}). "
            "JANGAN training. Cek template/format."
        )
    mo.md(
        f"✅ Masking benar pada {_n}/{_n} contoh. Rata-rata hanya **{sum(_fracs) / len(_fracs):.0%}** token yang dihitung loss-nya.\n\n"
        f"**Yang dilatih (contoh #0):**\n```\n{_sample}\n```"
    )
    return


@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="▶ Mulai training")
    train_btn
    return (train_btn,)


@app.cell
def _(mo, train_btn, trainer):
    mo.stop(not train_btn.value, mo.md("_Klik **Mulai training** di atas._"))
    train_result = trainer.train()
    _hist = trainer.state.log_history
    _evals = [(h["epoch"], h["eval_loss"]) for h in _hist if "eval_loss" in h]
    _txt = "\n".join(f"- epoch {e:.0f}: eval loss {v:.4f}" for e, v in _evals) or "- (tidak ada validasi)"
    mo.md(
        f"✅ Selesai. **Train loss akhir:** {train_result.training_loss:.4f}\n\n**Eval loss per epoch:**\n{_txt}\n\n"
        "Kalau eval loss naik di epoch terakhir → turunkan `EPOCHS` (atau tambah data yang beragam)."
    )
    return (train_result,)


@app.cell
def _(FastLanguageModel, model, text_tok):
    import torch as _torch

    def generate_reply(messages, max_new_tokens=600):
        """Generate dengan prompt inference yang SAMA PERSIS dengan format training."""
        FastLanguageModel.for_inference(model)
        _prompt = render_prompt(text_tok, messages)
        _inputs = text_tok(_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with _torch.no_grad():
            _out = model.generate(
                **_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                use_cache=True,
            )
        return text_tok.decode(_out[0][_inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    return (generate_reply,)


@app.cell
def _(mo):
    eval_btn = mo.ui.run_button(label="🧪 Evaluasi format & kedalaman di data validasi")
    eval_btn
    return (eval_btn,)


@app.cell
def _(
    EVAL_N,
    card_text,
    eval_btn,
    generate_reply,
    jd_bank,
    mo,
    profile_text,
    train_result,
    val_rows,
):
    mo.stop(not eval_btn.value, mo.md("_Klik tombol evaluasi setelah training selesai._"))
    mo.stop(not val_rows, mo.md("Dataset terlalu kecil: tidak ada data validasi."))
    _ = train_result  # pastikan urutan: setelah training

    _eval_rows = val_rows[:EVAL_N]
    _vocab_base = vocab_from(profile_text, card_text)
    _ok, _sims, _show = 0, [], []
    _subst, _deep, _words_all = 0, 0, []
    for _r in _eval_rows:
        _msgs = expand_turns(conversation_messages(_r, profile_text, card_text, jd_bank))[-1]
        _pred = generate_reply(_msgs[:-1])
        _secs, _why = parse_sections(_pred)
        _ok += _secs is not None
        _user = _msgs[-2]["content"]
        _s1 = _secs[SECTION_TAGS[0]] if _secs else ""
        _sims.append(difflib.SequenceMatcher(None, _user.lower(), _s1.lower()).ratio())
        if _secs:
            _s3 = _secs[SECTION_TAGS[2]]
            _nw = len(re.findall(r"[A-Za-z']+", _s3))
            _words_all.append(_nw)
            if _r.get("category") in SUBSTANTIVE:
                _subst += 1
                _jd = row_jd(_r, jd_bank) if profile_text.strip() else None
                _ov = overlap_count(_s3, _vocab_base | vocab_from(_jd))
                _deep += (_nw >= 70 and _ov >= 2)  # proxy kasar kedalaman: cukup panjang + memakai detail profil
        if len(_show) < 3:
            _show.append((_user, _pred, _msgs[-1]["content"], _why))

    _detail = "\n\n".join(
        f"**Q:** {u}\n\n**Prediksi:**\n```\n{p}\n```\n**Referensi:**\n```\n{r}\n```\nFormat: {'OK' if w is None else 'GAGAL - ' + w}"
        for u, p, r, w in _show
    )
    _avg_w = sum(_words_all) / len(_words_all) if _words_all else 0
    mo.md(
        f"**Format 4-bagian valid:** {_ok}/{len(_eval_rows)} ({_ok / len(_eval_rows):.0%})\n\n"
        f"**Kemiripan [1. TRANSKRIP] dengan ucapan asli:** {sum(_sims) / len(_sims):.0%}\n\n"
        f"**Rata-rata kata bagian 3:** {_avg_w:.0f}\n\n"
        f"**Jawaban substantif yang 'dalam'** (>= 70 kata dan >= 2 detail dari profil/kartu/lowongan; proxy kasar, bukan pengganti membaca): "
        f"{_deep}/{_subst}\n\n{_detail}"
    )
    return


@app.cell
def _(mo):
    probe_text = mo.ui.text_area(
        value="Why do you want to work with us?",
        label="Uji bebas: ketik ucapan pewawancara (bahasa Inggris)",
        full_width=True,
    )
    probe_jd = mo.ui.text_area(
        value="",
        label="LOWONGAN untuk uji (opsional; kosong = '(kosong)')",
        full_width=True,
    )
    probe_btn = mo.ui.run_button(label="Tanya")
    mo.vstack([probe_text, probe_jd, probe_btn])
    return probe_btn, probe_jd, probe_text


@app.cell
def _(
    card_text,
    generate_reply,
    mo,
    probe_btn,
    probe_jd,
    probe_text,
    profile_text,
    train_result,
):
    mo.stop(not probe_btn.value, mo.md("_Ketik pertanyaan lalu klik **Tanya**._"))
    _ = train_result
    _system = build_system_prompt(profile_text, card_text, probe_jd.value)
    _reply = generate_reply(
        [
            {"role": "system", "content": _system},
            {"role": "user", "content": probe_text.value},
        ]
    )
    _secs, _why = parse_sections(_reply)
    mo.md(f"```\n{_reply}\n```\nFormat: {'✅ OK' if _secs else '❌ ' + str(_why)}")
    return


@app.cell
def _(mo):
    save_btn = mo.ui.run_button(label="💾 Simpan LoRA adapter")
    hf_repo = mo.ui.text(label="HF repo (opsional)", placeholder="username/asisten-cv-lora")
    hf_token = mo.ui.text(label="HF token (opsional)", kind="password")
    mo.vstack([save_btn, hf_repo, hf_token])
    return hf_repo, hf_token, save_btn


@app.cell
def _(LORA_DIR, hf_repo, hf_token, mo, model, save_btn, tokenizer, train_result):
    mo.stop(not save_btn.value, mo.md("_Klik **Simpan LoRA adapter** setelah training._"))
    _ = train_result
    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    _msg = f"Tersimpan di `{LORA_DIR}/` (di dalam sandbox Molab; sesi bisa berakhir, jadi sebaiknya upload ke HF)."
    if hf_repo.value and hf_token.value:
        model.push_to_hub_merged(hf_repo.value, tokenizer, save_method="lora", token=hf_token.value)
        _msg += f" ✅ Di-upload ke `{hf_repo.value}`."
    mo.md(_msg)
    return


@app.cell
def _(mo):
    quants = mo.ui.multiselect(
        options=["q8_0", "q5_k_m", "q4_k_m"],
        value=["q8_0", "q4_k_m"],
        label="Kuantisasi GGUF",
    )
    gguf_btn = mo.ui.run_button(label="📦 Export GGUF")
    mo.vstack([quants, gguf_btn])
    return gguf_btn, quants


@app.cell
def _(GGUF_DIR, gguf_btn, mo, model, quants, tokenizer, train_result):
    mo.stop(not gguf_btn.value, mo.md("_Pilih kuantisasi lalu klik **Export GGUF** (butuh beberapa menit; Unsloth meng-clone dan membangun llama.cpp)._"))
    _ = train_result
    for _q in quants.value:
        model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method=_q)
    _files = sorted(Path(".").rglob("*.gguf"))
    mo.md("**File GGUF:**\n" + "\n".join(f"- `{p}` ({p.stat().st_size / 1e9:.2f} GB)" for p in _files))
    return


@app.cell
def _(card_text, mo, profile_text, text_tok):
    # Simpan kontrak inference supaya app live memakai susunan yang SAMA dengan training.
    Path("system_prompt_base.txt").write_text(build_system_prompt(profile_text, card_text, None), encoding="utf-8")
    _example = render_prompt(
        text_tok,
        [
            {"role": "system", "content": "<SYSTEM PROMPT: PROFIL + KARTU + LOWONGAN>"},
            {"role": "user", "content": "<ucapan pewawancara dari Whisper>"},
        ],
    )
    Path("inference_prompt_example.txt").write_text(_example, encoding="utf-8")
    mo.md(
        "### Kontrak format inference\n"
        "`system_prompt_base.txt` = system prompt training tanpa lowongan. Saat live pakai `copilot_prompt.py` "
        "(`load_system_prompt()` membaca `cv_profile.txt`, `knowledge_card.txt`, dan `lowongan.txt` hari-H). "
        "**Jangan taruh SYSTEM di Modelfile**; kirim lewat messages. Bentuk prompt yang dilihat model saat training dan "
        "harus diterimanya saat live:\n"
        f"```\n{_example}\n```\n"
        "Perhatikan akhirannya: blok `<think>` **kosong** sudah tertutup. Kalau runtime (Ollama/llama.cpp) "
        "tidak menghasilkan prompt persis seperti ini, jawaban model bisa kacau. Atur `num_ctx` sekitar 6144."
    )
    return


if __name__ == "__main__":
    app.run()
