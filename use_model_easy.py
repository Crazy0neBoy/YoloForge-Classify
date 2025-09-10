"""
use_model_easy.py
Простейший скрипт для использования обученной классификационной модели YOLOv8.
-------------------------------------------------------------------------------
Функции:
- Загружает веса модели (.pt)
- Принимает путь к файлу или папке с изображениями
- Делает предсказания и выводит top-1 класс + уверенность
"""

import os
from pathlib import Path
from ultralytics import YOLO


# ================== НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ ==================
WEIGHTS = r"models\cat&dog_exp\weights\best.pt"  # путь к весам модели
INPUT_PATH = r"data\cat&dog\cat"                     # путь к изображению или папке
IMAGE_SIZE = 224                                     # размер входного изображения
CONF_THRESHOLD = 0.75                                # минимальная уверенность
# =============================================================


def list_images(path):
    """Собирает список изображений по пути (файл или папка)."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    path = Path(path)
    if path.is_file():
        return [str(path)] if path.suffix.lower() in exts else []
    elif path.is_dir():
        return [str(p) for p in path.glob("**/*") if p.suffix.lower() in exts]
    else:
        return []


def main():
    # Загружаем модель
    model = YOLO(WEIGHTS)
    print(f"[INFO] Загружены веса: {WEIGHTS}")

    # Собираем входные изображения
    images = list_images(INPUT_PATH)
    if not images:
        print("[ERR] Нет подходящих изображений по указанному пути")
        return

    # Прогон инференса
    for img_path in images:
        results = model(img_path, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False)
        probs = results[0].probs  # вероятности по классам
        top1 = int(probs.top1)
        top1_conf = float(getattr(probs, "top1conf", 0.0))
        label = model.names[top1]
        print(f"{img_path} -> {label} ({top1_conf:.2f})")


if __name__ == "__main__":
    main()