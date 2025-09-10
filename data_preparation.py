import os
import shutil
import random
import hashlib
import json
import cv2
from PIL import Image
import albumentations as A
from tqdm import tqdm

# ===================== НАСТРОЙКИ =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Имя модели (меняй здесь для переключения)
model_name = "cat&dog"

# Пути
source_dir = os.path.join(BASE_DIR, "data", model_name)
target_dir = os.path.join(BASE_DIR, "datasets", model_name)
config_path = os.path.join(source_dir, "config.json")
hash_file_path  = os.path.join(source_dir, "hashes.json")
split_map_path  = os.path.join(source_dir, "split_map.json")

# Конфиг по умолчанию
default_config = {
    "model_name": model_name,  # Имя текущей модели (папки в data/ и datasets/)
    # Разбиение
    "train_ratio": 0.6,
    "valid_ratio": 0.25,
    "test_ratio": 0.15,
    "image_size": [224, 224],  # Размер изображения (классификация YOLOv8 с 224x224)
    # Аугментация
    "augmentation_factor": 1,             # 0 = полное выключение
    "enable_noise_artifacts": True,       # Шум
    "enable_geometric_transforms": True,  # Геометрия
    # Принудительное полное обновление (игнорировать хэши)
    "force_rebuild": False
}

def ensure_initial_structure():
    """Создаём базовую структуру и конфиг, если модель ещё не создана."""
    if not os.path.exists(source_dir):
        os.makedirs(os.path.join(source_dir, "class1"), exist_ok=True)
        os.makedirs(os.path.join(source_dir, "class2"), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Создана структура для модели '{model_name}':")
        print("  ", os.path.join(source_dir, "class1"))
        print("  ", os.path.join(source_dir, "class2"))
        print("  ", config_path)
        print("\n[INFO] Добавь изображения и при необходимости измени config.json, затем запусти снова.")
        exit(0)

def load_config():
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Создан {config_path}. Отредактируй при необходимости и запусти снова.")
        exit(0)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

# ===================== ВСПОМОГАТЕЛЬНОЕ =====================
def file_md5(path, block_size=65536):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def list_classes():
    return [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

def scan_source_hashes(classes):
    all_hashes = {}
    for cls in classes:
        class_source = os.path.join(source_dir, cls)
        files = [
            f for f in os.listdir(class_source)
            if os.path.isfile(os.path.join(class_source, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        class_hashes = {}
        for fname in files:
            class_hashes[fname] = file_md5(os.path.join(class_source, fname))
        all_hashes[cls] = class_hashes
    return all_hashes

def make_dirs_for_classes(classes):
    for split in ("train", "valid", "test"):
        for cls in classes:
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

def remove_aug_variants(split_dir, base_name):
    for f in os.listdir(split_dir):
        if f.startswith(base_name + "_augm"):
            try:
                os.remove(os.path.join(split_dir, f))
            except Exception as e:
                print(f"[WARN] Не удалось удалить {f}: {e}")

def copy_and_resize(src, dst, image_size):
    with Image.open(src) as im:
        im = im.convert("RGB")
        im = im.resize(image_size, Image.Resampling.LANCZOS)
        im.save(dst)

def compute_target_counts_from_split_map(split_map, classes):
    counts = {s: {cls:0 for cls in classes} for s in ("train", "valid", "test")}
    for key, split in split_map.items():
        if "/" in key:
            cls, _ = key.split("/", 1)
            if cls in classes and split in counts:
                counts[split][cls] += 1
    return counts

def choose_split_for_new(cls, counts, ratios):
    total = sum(counts[s][cls] for s in ("train","valid","test")) + 1e-9
    current = {
        "train": counts["train"][cls] / total,
        "valid": counts["valid"][cls] / total,
        "test":  counts["test"][cls]  / total
    }
    score  = {s: (current[s] / (ratios[s] + 1e-9)) for s in current}
    return min(score, key=score.get)

# ===================== СПЛИТ И ПОЛНЫЙ РЕБИЛД =====================
def full_rebuild(new_hashes, classes, cfg):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        print(f"[INFO] Полное удаление {target_dir} (force_rebuild=True).")
    make_dirs_for_classes(classes)

    all_items = []
    for cls, files in new_hashes.items():
        for fname in files.keys():
            all_items.append((cls, fname))
    random.shuffle(all_items)

    split_map = {}
    class_counts = {s:{c:0 for c in classes} for s in ("train","valid","test")}
    ratios = {"train":cfg["train_ratio"], "valid":cfg["valid_ratio"], "test":cfg["test_ratio"]}

    for cls, fname in all_items:
        split = choose_split_for_new(cls, class_counts, ratios)
        class_counts[split][cls] += 1
        split_map[f"{cls}/{fname}"] = split

        src = os.path.join(source_dir, cls, fname)
        dst = os.path.join(target_dir, split, cls, fname)
        try:
            copy_and_resize(src, dst, tuple(cfg["image_size"]))
        except Exception as e:
            print(f"[ERR] Копирование {src} → {dst} провалилось: {e}")

    return split_map

# ===================== ИНКРЕМЕНТАЛЬНАЯ СИНХРОНИЗАЦИЯ =====================
def incremental_sync(old_hashes, new_hashes, split_map, classes, cfg):
    make_dirs_for_classes(classes)

    removed, changed_or_new = [], []

    for cls, old_files in old_hashes.items():
        for fname in old_files:
            if fname not in new_hashes.get(cls, {}):
                removed.append(f"{cls}/{fname}")

    counts = compute_target_counts_from_split_map(split_map, classes)

    for cls, files in new_hashes.items():
        old_files = old_hashes.get(cls, {})
        for fname, h in files.items():
            if old_files.get(fname) != h:
                changed_or_new.append((cls, fname))

    if not removed and not changed_or_new:
        print(f"[INFO] Изменений в данных для модели '{cfg['model_name']}' не обнаружено.")
        return split_map, False

    for key in removed:
        split = split_map.get(key)
        cls, fname = key.split("/", 1)
        if split:
            base = os.path.join(target_dir, split, cls)
            path = os.path.join(base, fname)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    remove_aug_variants(base, os.path.splitext(fname)[0])
                    print(f"[INFO] Удалён: {path}")
                except Exception as e:
                    print(f"[WARN] Не удалось удалить {path}: {e}")
            split_map.pop(key, None)

    if changed_or_new:
        print(f"[INFO] Обновляем/добавляем {len(changed_or_new)} изображений...")
        ratios = {"train":cfg["train_ratio"], "valid":cfg["valid_ratio"], "test":cfg["test_ratio"]}
        for cls, fname in tqdm(changed_or_new, desc="Sync"):
            key = f"{cls}/{fname}"
            split = split_map.get(key)
            if split is None:
                split = choose_split_for_new(cls, counts, ratios)
                counts[split][cls] += 1
                split_map[key] = split
            src = os.path.join(source_dir, cls, fname)
            dst_dir = os.path.join(target_dir, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fname)
            try:
                copy_and_resize(src, dst, tuple(cfg["image_size"]))
            except Exception as e:
                print(f"[ERR] Копирование {src} → {dst} провалилось: {e}")

    return split_map, True

# ===================== АУГМЕНТАЦИИ =====================
def build_gauss_noise():
    try:
        return A.GaussNoise(var_limit=(10.0, 50.0), p=1)
    except TypeError:
        return A.GaussNoise(p=1)

def augment_dataset(classes, cfg, available_transforms):
    if cfg["augmentation_factor"] <= 0 or not available_transforms:
        print("[INFO] Аугментация отключена.")
        return

    base_images = []
    for cls in classes:
        split_path = os.path.join(target_dir, "train", cls)
        if not os.path.isdir(split_path):
            continue
        files = [
            f for f in os.listdir(split_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            and "_augm" not in f.lower()
        ]
        base_images += [(split_path, f) for f in files]

    if not base_images:
        print("[INFO] Нет изображений для аугментации.")
        return

    for cls in classes:
        split_path = os.path.join(target_dir, "train", cls)
        if not os.path.isdir(split_path):
            continue
        for f in os.listdir(split_path):
            if "_augm" in f.lower():
                try:
                    os.remove(os.path.join(split_path, f))
                except Exception as e:
                    print(f"[WARN] Не удалён {f}: {e}")

    with tqdm(total=len(base_images) * cfg["augmentation_factor"], desc="Augment") as pbar:
        for split_path, fname in base_images:
            img_path = os.path.join(split_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Не удалось открыть {img_path}")
                continue
            for i in range(cfg["augmentation_factor"]):
                transform = random.choice(available_transforms)
                augmented = transform(image=img)["image"]
                base, ext = os.path.splitext(fname)
                out_name = f"{base}_augm{i+1}{ext}"
                cv2.imwrite(os.path.join(split_path, out_name), augmented)
                pbar.update(1)

# ===================== СТАТИСТИКА =====================
def compute_stats():
    stats = {"train":{}, "valid":{}, "test":{}}
    for split in stats.keys():
        split_dir = os.path.join(target_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            n = sum(1 for f in os.listdir(cls_dir) if f.lower().endswith((".png",".jpg",".jpeg",".bmp")))
            stats[split][cls] = n
    return stats

def print_stats(stats):
    print("\n=== Статистика датасета ===")
    grand_total = 0
    for split in ("train", "valid", "test"):
        print(f"\n{split.capitalize()}:")
        total = sum(stats.get(split, {}).values())
        for cls, cnt in sorted(stats.get(split, {}).items()):
            print(f"  {cls}: {cnt} изображений")
        print(f"  Всего: {total} изображений")
        grand_total += total
    print(f"\n[INFO] Общий итог: {grand_total} изображений")

# ===================== MAIN =====================
if __name__ == "__main__":
    ensure_initial_structure()
    cfg = load_config()

    classes = list_classes()
    if not classes:
        print(f"[WARN] В {source_dir} нет подпапок-классов. Создай, например, class1, class2.")
        exit(0)

    new_hashes = scan_source_hashes(classes)

    if cfg["force_rebuild"]:
        split_map = full_rebuild(new_hashes, classes, cfg)
        save_json(split_map_path, split_map)
        save_json(hash_file_path, new_hashes)
        changed = True
    else:
        old_hashes = load_json(hash_file_path, {})
        split_map  = load_json(split_map_path,  {})
        split_map, changed = incremental_sync(old_hashes, new_hashes, split_map, classes, cfg)
        save_json(split_map_path, split_map)
        save_json(hash_file_path, new_hashes)

    # Список трансформаций
    noise_artifacts = [
        build_gauss_noise(),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
        A.MotionBlur(blur_limit=7, p=1),
        A.ImageCompression(quality_range=(30, 70), p=1),
    ]
    geometric_transforms = [
        A.HorizontalFlip(p=1),
        A.RandomRotate90(p=1),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-30, 30), p=1),
        A.OpticalDistortion(distort_limit=0.05, p=1),
    ]
    available_transforms = []
    if cfg["enable_noise_artifacts"]:
        available_transforms.extend(noise_artifacts)
    if cfg["enable_geometric_transforms"]:
        available_transforms.extend(geometric_transforms)

    # Аугментация
    if changed and cfg["augmentation_factor"] > 0 and available_transforms:
        print(f"\n[INFO] Приступаем к аугментации... (factor={cfg['augmentation_factor']})")
        augment_dataset(classes, cfg, available_transforms)
    else:
        print("\n[INFO] Аугментация пропущена.")

    if changed:
        stats = compute_stats()
        print_stats(stats)
        print(f"\n[INFO] Датасет модели '{cfg['model_name']}' обновлён в: {target_dir}")
    else:
        print(f"\n[INFO] Изменений в данных для модели '{cfg['model_name']}' не обнаружено.")