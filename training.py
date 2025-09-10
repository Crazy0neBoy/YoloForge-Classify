import os
import json
import torch
from ultralytics import YOLO
import logging

# Настройка логирования (INFO только один раз)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("training")

# ========== Настройки ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Имя модели (должно совпадать с папкой в data/<model_name>)
model_name = "cat&dog"

# Пути
source_dir = os.path.join(BASE_DIR, "data", model_name)
config_path = os.path.join(source_dir, "config.json")
dataset_path = os.path.join(BASE_DIR, "datasets", model_name)

# Загрузка конфига
if not os.path.exists(config_path):
    raise FileNotFoundError(f"[ERR] Файл конфигурации {config_path} не найден. Сначала запусти data_preparation.py.")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

# Параметры из конфига
imgsz = cfg.get("image_size", [224, 224])[0]
model_old = r'models\yolov8s-cls.pt'   # Можно заменить на best.pt предыдущей модели
model_name_for_save = f"{cfg['model_name']}_exp"

# Проверка датасета
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Папка датасета {dataset_path} не найдена. Сначала запусти data_preparation.py.")

# Загружаем модель
try:
    model = YOLO(model_old)
    logger.info(f"Модель {model_old} успешно загружена.")
except Exception as e:
    logger.error(f"Ошибка при загрузке модели: {str(e)}")
    raise

# Устройство
device = 0 if torch.cuda.is_available() else 'cpu'
logger.info(f"Используемое устройство: {'GPU' if device == 0 else 'CPU'}")

# ========== Тренировка ==========
if __name__ == '__main__':
    try:
        results = model.train(
            data=dataset_path,      # Путь к датасету (train/valid/test)
            model=model_old,        # Предобученные веса
            epochs=cfg.get("epochs", 60),  # Количество эпох обучения
            imgsz=imgsz,            # Размер входных изображений
            batch=-1,               # Автоподбор размера батча
            device=device,          # Устройство (0 — GPU, 'cpu' — CPU)
            name=model_name_for_save, # Имя эксперимента (папка сохранения)
            exist_ok=True,          # Разрешить перезапись эксперимента

            # ==== Аугментации ====
            hsv_h=0.02,             # Сдвиг оттенка (±2%)
            hsv_s=0.5,              # Сдвиг насыщенности (±50%)
            hsv_v=0.3,              # Сдвиг яркости (±30%)
            fliplr=0.5,             # Горизонтальное отражение (50%)
            flipud=0.1,             # Вертикальное отражение (10%)
            translate=0.1,          # Сдвиг изображения (±10%)
            scale=0.35,             # Масштабирование (±35%)
            shear=2.0,              # Сдвиг/срез (±2°)
            perspective=0.0001,     # Перспективное искажение
            mosaic=1.0,             # Мозаика (смешивание картинок)
            mixup=0.2,              # MixUp (смешивание картинок и меток)

            # ==== Оптимизация ====
            optimizer='AdamW',      # Оптимизатор AdamW
            lr0=0.001,              # Начальная скорость обучения
            patience=10,            # Раннее прекращение при отсутствии прогресса
            weight_decay=0.0005,    # L2-регуляризация (борьба с переобучением)
            dropout=0.1,            # Dropout в модели
            save=True,              # Сохранять чекпоинты
            save_period=0,          # Сохранять каждые 0 эпох
            pretrained=True,        # Использовать предобученные веса
            verbose=False,          # Подробный вывод в консоль (выкл)
            project='models',       # Папка для сохранения результатов
            seed=42                 # Фиксируем случайность
        )
        logger.info("Обучение завершено.")

        # Валидация
        val_results = model.val(data=dataset_path)
        logger.info(f"Валидация: Top-1 Accuracy={val_results.top1:.4f}, Top-5 Accuracy={val_results.top5:.4f}")

        # Классы
        logger.info(f"Классы: {model.names}")

        # Сохранение
        save_dir = os.path.join('models', model_name_for_save)
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, 'best.pt'))
        logger.info(f"Модель сохранена в {save_dir}/best.pt")

    except Exception as e:
        logger.error(f"Ошибка во время тренировки или валидации: {str(e)}")
        raise