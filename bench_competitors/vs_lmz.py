"""results.json -> bench_competitors/VS_LMZ.md: Seedpack against lmz on every dataset."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "results.json")))

SINGLE = [("bf16-0.6b", "BF16 Qwen3-0.6B"), ("bf16-2b", "BF16 Qwen3.5-2B"), ("q8_0-0.6b", "Q8_0 Qwen3-0.6B, один файл"),
          ("q4_k_m-0.6b", "Q4_K_M Qwen3-0.6B, один файл"), ("repo-0.6b", "Репозиторий Qwen3-0.6B (32 файла)"),
          ("repo-0.8b", "Репозиторий Qwen3.5-0.8B (28 файлов)"), ("repo-2b", "Репозиторий Qwen3.5-2B (28 файлов)")]
QUANTS = [("quants-0.6b", "Qwen3-0.6B, 25 квантов"), ("quants-0.8b", "Qwen3.5-0.8B, 21 квант"),
          ("quants-2b", "Qwen3.5-2B, 21 квант"), ("quants-le4-0.6b", "Qwen3-0.6B, 17 квантов не старше Q4"),
          ("quantsim-0.6b", "Qwen3-0.6B, 25 квантов + imatrix"), ("quantsim-0.8b", "Qwen3.5-0.8B, 21 квант + imatrix"),
          ("quantsim-2b", "Qwen3.5-2B, 21 квант + imatrix")]


def gb(b):
    return f"{b / 1e9:.2f} ГБ" if b >= 1e9 else f"{b / 1e6:.1f} МБ"


def sec(n, s):
    return f"{s:.1f} с ({n / s / 1e6:,.0f} МБ/с)".replace(",", " ") if s else "—"


def table(rows, title):
    out = [f"## {title}", "", "| Набор | Исходно | Seedpack | lmz | Seedpack / lmz | Сжатие Seedpack / lmz | Распаковка Seedpack / lmz |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for key, name in rows:
        s, l = r.get(f"{key} | SeedPack"), r.get(f"{key} | lmz 1.3.0")
        if not s or not l or "archive" not in s or "archive" not in l:
            continue
        n = s["size"]
        out.append(f"| {name} | {gb(n)} | **{gb(s['archive'])} ({100 * s['archive'] / n:.2f}%)** | {gb(l['archive'])} "
                   f"({100 * l['archive'] / n:.2f}%) | {s['archive'] / l['archive']:.3f} | "
                   f"{s['compress_s']:.1f} с / {l['compress_s']:.1f} с | {s['decompress_s']:.1f} с / {l['decompress_s']:.1f} с |")
    return out + [""]


lines = ["# Seedpack vs lmz 1.3.0", "",
         "Одна машина (RTX 5070, Core i7-8700K (6 ядер, 12 потоков), NVMe). Сжатие — от файлов на диске до архива; распаковка — декодирование "
         "с проверкой без записи на диск (`seedpack test` / `lmz verify`). Время Seedpack включает запуск процесса и "
         "инициализацию GPU (2–4 с). Все архивы обоих инструментов восстанавливаются побайтово.", ""]
lines += table(SINGLE, "Одиночные файлы и репозитории целиком")
lines += table(QUANTS, "Наборы квантов без BF16")
lines += ["Наборы квантов: все GGUF-кванты репозитория (IQ1…Q8_0, UD-варианты) без BF16-модели и mmproj; «+ imatrix» — "
          "вместе с файлом imatrix репозитория. Seedpack берёт старший квант как семя и выводит из него остальные.", ""]
open(os.path.join(HERE, "VS_LMZ.md"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
