# use_model.py
# Универсальный и дружелюбный скрипт для использования обученной классификационной модели YOLOv8.
# -----------------------------------------------------------------------------
# Возможности:
#  - загрузка конфигурации из data/<model_name>/config.json (imgsz, имя модели и др.)
#  - автоматический поиск "последних" весов в models/<model_name>_exp*/weights/best.pt
#  - вход: одиночный файл, папка (с опцией --recursive), список из .txt
#  - батчевый инференс, порог уверенности, top-K
#  - экшены по классам: ignore / move / copy / delete (+ dry-run)
#  - CSV-отчёт с результатами
#  - минимальный и читаемый вывод, без логгеров, только print и прогресс-бар
#
# Примеры:
#   python use_model.py --model-name "cat&dog" --input "D:/images" --recursive --conf 0.9 --batch 64
#   python use_model.py --model-name "cat&dog" --weights "models/cat&dog_exp/weights/best.pt" --input img.jpg
#   python use_model.py --model-name "cat&dog" --input list.txt --actions actions.json --output-dir "_sort" --dry-run
#
# Формат actions.json:
# {
#   "action_map": { "cat": "move", "dog": "ignore" },
#   "dest_root": "D:/sorted",         # куда перемещать/копировать
#   "unknown_action": "ignore"         # что делать с неизвестными классами
# }
#
# Примечания:
#  - Для классификации Ultralytics инференс можно вызывать как model(paths, ...)
#  - Имя модели можно оставить только через --weights (без --model-name), но тогда imgsz берётся из аргумента
# -----------------------------------------------------------------------------

import os
import sys
import json
import csv
import glob
import time
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from tqdm import tqdm
from ultralytics import YOLO


# ----------------------------- Утилиты путей -----------------------------
def to_abs(path: str) -> str:
    """Превращает путь в абсолютный (безопасно для Windows/Linux)."""
    return str(Path(path).expanduser().resolve())


def list_images_in_dir(root: str, recursive: bool, exts: Tuple[str, ...]) -> List[str]:
    """Собирает все изображения в папке (при recursive=True — во всех подпапках)."""
    root = to_abs(root)
    patterns = [f"**/*{e}" if recursive else f"*{e}" for e in exts]
    paths = []
    for p in patterns:
        paths.extend([str(Path(x)) for x in glob.glob(str(Path(root) / p), recursive=recursive)])
    return sorted(list(dict.fromkeys(paths)))  # уникализируем порядок


def read_list_file(txt_path: str) -> List[str]:
    """Читает .txt со списком путей (по одному на строку)."""
    txt_path = to_abs(txt_path)
    out = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                out.append(to_abs(p))
    return out


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------- Загрузка конфига ---------------------------
def load_model_config(base_dir: str, model_name: Optional[str]) -> Dict:
    """
    Загружает config.json из data/<model_name>/, если model_name задан.
    Если же model_name=None, вернёт пустой конфиг (параметры нужно задавать флагами).
    """
    if not model_name:
        return {}
    cfg_path = Path(base_dir) / "data" / model_name / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"[ERR] Не найден конфиг: {cfg_path}. Сначала подготовь данные и конфиг.")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------- Поиск последних весов модели --------------------
def guess_latest_weights(models_dir: str, model_name: str) -> Optional[str]:
    """
    Ищет последние веса вида models/<model_name>_exp*/weights/best.pt по времени модификации.
    Возвращает путь или None, если не найдено.
    """
    pattern = str(Path(models_dir) / f"{model_name}_exp*/weights/best.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    # сортируем по времени изменения весов
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return to_abs(candidates[0])


# ----------------------------- Экшены по файлам --------------------------
def safe_move(src: str, dst: str) -> None:
    ensure_dir(os.path.dirname(dst))
    shutil.move(src, dst)


def safe_copy(src: str, dst: str) -> None:
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def apply_action(
    action: str,
    image_path: str,
    label: str,
    dest_root: Optional[str],
    dry_run: bool
) -> Optional[str]:
    """
    Выполняет действие над файлом:
      - "ignore": ничего
      - "move":   переместить в <dest_root>/<label>/
      - "copy":   копировать   в <dest_root>/<label>/
      - "delete": удалить файл
    Вернёт путь назначения при move/copy, иначе None.
    """
    action = (action or "ignore").lower()
    if action == "ignore":
        return None
    if action in ("move", "copy"):
        if not dest_root:
            # без dest_root переносить/копировать некуда — игнорируем
            return None
        dst = str(Path(dest_root) / label / Path(image_path).name)
        if dry_run:
            return dst
        (safe_move if action == "move" else safe_copy)(image_path, dst)
        return dst
    if action == "delete":
        if dry_run:
            return None
        try:
            os.remove(image_path)
        except OSError:
            pass
        return None
    # неизвестное действие — игнор
    return None


# ----------------------------- CSV отчёт ---------------------------------
def write_csv(rows: List[Dict], csv_path: str) -> None:
    """Сохраняет CSV с полями: path,label,conf,second_label,second_conf"""
    ensure_dir(os.path.dirname(csv_path))
    fieldnames = ["path", "label", "conf", "second_label", "second_conf", "action", "dest"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ------------------------------- main ------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Гибкий инференс классификационной модели YOLOv8 (классы, батчи, экшены, отчёты)."
    )

    # Базовые пути/модель
    parser.add_argument("--model-name", type=str, default=None,
                        help="Имя модели (папка в data/<model_name> и models/<model_name>_exp*). "
                             "Если не указано — конфиг не читается и нужно задать --weights и --imgsz.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Путь к весам .pt. Если не задано — будет поиск в models/<model_name>_exp*/weights/best.pt.")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Каталог с экспериментами (по умолчанию: models).")

    # Источник данных
    parser.add_argument("--input", type=str, required=True,
                        help="Путь к файлу ИЛИ папке ИЛИ .txt со списком путей.")
    parser.add_argument("--recursive", action="store_true",
                        help="Рекурсивный проход по подпапкам (для --input=папка).")
    parser.add_argument("--exts", type=str, default=".jpg,.jpeg,.png,.bmp,.webp",
                        help="Список расширений через запятую (для обхода папок).")

    # Инференс
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Размер входного изображения. "
                             "Если не задан — возьмём из config.json (когда есть --model-name) "
                             "или 224 по умолчанию.")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Порог уверенности (для принятия действия).")
    parser.add_argument("--batch", type=int, default=32,
                        help="Размер батча при инференсе.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Устройство: 'auto' | 'cpu' | '0' (GPU номер).")

    # Экшены и отчёт
    parser.add_argument("--actions", type=str, default=None,
                        help="JSON-файл с полями: { action_map: {label: action}, dest_root: str, unknown_action: str }")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Куда перемещать/копировать (если action=move/copy). Можно оставить пустым и задать в actions.json как dest_root.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Режим прогона без изменений на диске (ничего не удаляем/не переносим).")
    parser.add_argument("--csv", type=str, default=None,
                        help="Путь для CSV-отчёта. По умолчанию в ./outputs/<model_name>_<timestamp>.csv")

    # Вывод/поведение
    parser.add_argument("--topk", type=int, default=2,
                        help="Сколько лучших классов писать в отчёт (1..5).")
    parser.add_argument("--verbose", action="store_true",
                        help="Болтать больше в консоль (выводить каждый файл).")

    args = parser.parse_args()

    BASE_DIR = str(Path(__file__).resolve().parent)
    cfg = load_model_config(BASE_DIR, args.model_name) if args.model_name else {}

    # img size
    imgsz = args.imgsz or (cfg.get("image_size", [224, 224])[0] if cfg else 224)
    # device
    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # weights
    weights = args.weights
    if not weights:
        if not args.model_name:
            raise ValueError("Не указаны --weights и --model-name. Нужен хотя бы один из них.")
        weights = guess_latest_weights(Path(BASE_DIR) / args.models_dir, args.model_name)
        if not weights:
            raise FileNotFoundError(
                f"Не удалось найти веса для модели '{args.model_name}' в '{args.models_dir}'. "
                f"Укажи --weights вручную."
            )
    weights = to_abs(weights)

    # собрать входные пути
    exts = tuple(e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower()
                 for e in args.exts.split(",") if e.strip())
    input_path = to_abs(args.input)
    inputs: List[str] = []

    if os.path.isfile(input_path):
        if input_path.lower().endswith(".txt"):
            inputs = read_list_file(input_path)
        elif input_path.lower().endswith(exts):
            inputs = [input_path]
        else:
            raise ValueError(f"[ERR] Неподдерживаемый тип входного файла: {input_path}")
    elif os.path.isdir(input_path):
        inputs = list_images_in_dir(input_path, args.recursive, exts)
    else:
        raise FileNotFoundError(f"[ERR] Входной путь не найден: {input_path}")

    if not inputs:
        print("[INFO] Нет подходящих входных изображений.")
        sys.exit(0)

    # actions.json
    action_map: Dict[str, str] = {}
    dest_root: Optional[str] = None
    unknown_action = "ignore"
    if args.actions:
        with open(to_abs(args.actions), "r", encoding="utf-8") as f:
            data = json.load(f)
        action_map = {str(k): str(v).lower() for k, v in data.get("action_map", {}).items()}
        dest_root = data.get("dest_root") or args.output_dir
        if dest_root:
            dest_root = to_abs(dest_root)
        unknown_action = data.get("unknown_action", "ignore")
    else:
        # если JSON не задан, но указан --output-dir, будем использовать его как dest_root,
        # а все классы — "ignore" (ничего не делаем), пока пользователь не задаст явный маппинг.
        dest_root = to_abs(args.output_dir) if args.output_dir else None

    # загрузка модели
    model = YOLO(weights)
    print(f"[INFO] Загружены веса: {weights}")
    print(f"[INFO] Устройство: {'GPU' if device == 0 else device}  |  imgsz={imgsz}  |  batch={args.batch}")

    # список меток из модели
    names = model.names  # dict: {0: "class_name", ...}
    idx_to_name = {int(k): v for k, v in names.items()} if isinstance(names, dict) else {i: n for i, n in enumerate(names)}

    # подготовка CSV файла
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = args.csv or to_abs(str(Path("outputs") / f"{args.model_name or 'model'}_{timestamp}.csv"))

    # батчевый прогон
    rows: List[Dict] = []
    total = len(inputs)
    batch = max(1, int(args.batch))

    def emit_verbose(msg: str):
        if args.verbose:
            print(msg)

    for i in tqdm(range(0, total, batch), desc="Inference"):
        batch_paths = inputs[i: i + batch]

        # В классификации YOLO можно передавать список путей сразу:
        # conf — порог уверенности; imgsz — размер; device — GPU/CPU; verbose=False — тихий режим
        results = model(batch_paths, conf=args.conf, imgsz=imgsz, device=device, verbose=False)

        for src_path, res in zip(batch_paths, results):
            # res.probs: вероятности по классам
            probs = res.probs
            if probs is None:
                # что-то пошло не так, пропускаем
                continue

            # top-1
            top1_idx = int(probs.top1)
            top1_conf = float(getattr(probs, "top1conf", 0.0))
            label = idx_to_name.get(top1_idx, str(top1_idx))

            # top-k (на случай, если пользователь хочет видеть второй лучший вариант)
            second_label, second_conf = "", 0.0
            if args.topk and args.topk > 1:
                # probs.data — вектор вероятностей (torch.Tensor)
                vec = probs.data if hasattr(probs, "data") else None
                if vec is not None:
                    k = min(args.topk, len(vec))
                    topk_conf, topk_idx = torch.topk(vec, k)
                    if k >= 2:
                        second_label = idx_to_name.get(int(topk_idx[1]), str(int(topk_idx[1])))
                        second_conf = float(topk_conf[1])

            # Решение по действию
            action = action_map.get(label, unknown_action)
            dest = apply_action(action, src_path, label, dest_root, args.dry_run)

            # Отчёт
            rows.append({
                "path": src_path,
                "label": label,
                "conf": f"{top1_conf:.4f}",
                "second_label": second_label,
                "second_conf": f"{second_conf:.4f}" if second_label else "",
                "action": action,
                "dest": dest or ""
            })

            # Подробный вывод (опционально)
            emit_verbose(f"{src_path}  ->  {label} ({top1_conf:.3f})  action={action}{' (dry-run)' if args.dry_run else ''}")

    # сохраняем CSV
    write_csv(rows, csv_path)
    print(f"[INFO] CSV-отчёт: {csv_path}")

    # финальная сводка
    # посчитаем по классам и по действиям
    by_label: Dict[str, int] = {}
    by_action: Dict[str, int] = {}
    decided = 0
    for r in rows:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1
        by_action[r["action"]] = by_action.get(r["action"], 0) + 1
        if float(r["conf"] or 0) >= args.conf:
            decided += 1

    print("\n=== Сводка ===")
    print(f"Всего обработано: {len(rows)}")
    print(f"С уверенностью >= {args.conf}: {decided}")
    if by_label:
        print("По классам:")
        for k in sorted(by_label):
            print(f"  {k}: {by_label[k]}")
    if by_action:
        print("По действиям:")
        for k in sorted(by_action):
            print(f"  {k}: {by_action[k]}")
    if args.dry_run:
        print("\n[NOTE] Был включён --dry-run: файлы НЕ перемещались и НЕ удалялись.")

if __name__ == "__main__":
    main()