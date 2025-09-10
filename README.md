# YoloForge-Classify

Универсальный, GPU-дружественный пайплайн для **классификации изображений** на базе Ultralytics YOLO* — от подготовки датасета и аугментаций до обучения и гибкого пакетного инференса с автоматической сортировкой/действиями и CSV-отчётами. Работает с любыми совместимыми весами классификации Ultralytics YOLO (например, v8 сегодня, v* завтра).

> **Коротко:** положи изображения в `data/<model_name>/<class>/`, запусти `data_preparation.py`, обучи модель через `training.py`, а потом классифицируй при помощи простого `use_model_easy.py` или расширенного `use_model_smart.py`. CUDA поддерживается, но не обязательна.

---

## Содержание
- [Зачем нужен проект](#зачем-нужен-проект)
- [Возможности](#возможности)
- [Структура репозитория](#структура-репозитория)
- [Установка (важен порядок)](#установка-важен-порядок)
- [Быстрый старт](#быстрый-старт)
  - [1) Подготовка данных](#1-подготовка-данных)
  - [2) Обучение](#2-обучение)
  - [3) Использование модели](#3-использование-модели)
- [Конфигурация](#конфигурация)
- [Аугментации](#аугментации)
- [Действия и CSV-отчёты](#действия-и-csv-отчёты)
- [Проверка CUDA](#проверка-cuda)
- [Устранение неполадок](#устранение-неполадок)
- [FAQ](#faq)
- [Благодарности](#благодарности)

---

## Зачем нужен проект
Большинство репозиториев «обучил и используй» жёстко завязаны на конкретную версию YOLO. **YoloForge-Classify** сосредоточен на *рабочем процессе*: чистая подготовка датасета с хэшированием и инкрементальной синхронизацией, воспроизводимое обучение и удобные скрипты инференса, которые могут перемещать/копировать/удалять файлы по предсказанным классам и экспортировать CSV-отчёты. Если Ultralytics сохранит стабильный API для классификации, можно будет использовать новые версии YOLO* без изменений.

> * Внутри используется интерфейс классификации Ultralytics.

---

## Возможности
- **Построитель датасета с хэшами и синхронизацией** — поддерживает зеркальную структуру `datasets/<model_name>/train|valid|test` по долям из `config.json`. Обнаруживает добавленные/изменённые/удалённые файлы; есть режим полной пересборки.
- **Автоматический ресайз** под `image_size` при экспорте.
- **Аугментации Albumentations** (группы включаются/выключаются, есть множитель) с сохранением копий в `train/` и суффиксом `_augm#`.
- **Воспроизводимость**: сплиты сохраняются в `split_map.json`, хэши — в `hashes.json`.
- **Скрипт обучения**: загружает базовые веса Ultralytics, тренирует, валидирует, сохраняет модель в `models/<model_name>_exp/`.
- **Два режима инференса**:
  - `use_model_easy.py` — редактируешь константы и запускаешь.
  - `use_model_smart.py` — мощный CLI: батчи, top-K, порог уверенности, выбор устройства, рекурсивный обход, список файлов из `.txt`, действия по классам (ignore/move/copy/delete), dry-run, CSV-отчёт, авто-поиск последних весов.
- **Поддержка CUDA** и отдельный скрипт проверки GPU.

---

## Структура репозитория
```
YoloForge-Classify/
├─ data/
│  └─ <model_name>/
│     ├─ <class1>/, <class2>/, ...
│     ├─ config.json
│     ├─ hashes.json
│     └─ split_map.json
├─ datasets/
│  └─ <model_name>/
│     ├─ train/<class>/...
│     ├─ valid/<class>/...
│     └─ test/<class>/...
├─ models/
│  └─ <model_name>_exp.../weights/best.pt
├─ utilities/
│  └─ CUDA_Test.py
├─ data_preparation.py
├─ training.py
├─ use_model_easy.py
└─ use_model_smart.py
```

---

## Установка (важен порядок)
> Рекомендуется Python **3.10**. Лучше использовать виртуальное окружение.

### Windows
```bat
python -m venv .venv
.\.venv\Scripts\activate
:: --- установка строго в этом порядке ---
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install numpy==1.26.4 opencv-python==4.8.0.76
:: дополнительные
pip install albumentations tqdm
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
# --- установка строго в этом порядке ---
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install numpy==1.26.4 opencv-python==4.8.0.76
# дополнительные
pip install albumentations tqdm
```

**Почему такой порядок?** Сначала ставится правильный wheel PyTorch (с поддержкой CUDA), затем Ultralytics (тянет совместимые пакеты), потом фиксированные NumPy/OpenCV. Albumentations и tqdm нужны для аугментаций и прогресса.

---

## Быстрый старт

### 1) Подготовка данных
1. Создай модель, например `cat&dog`.
2. Положи исходные изображения в `data/cat&dog/<class>/...`.
3. Запусти:
   ```bash
   python data_preparation.py
   ```

Что произойдёт:
- вычислятся хэши файлов, создастся/обновится `datasets/<model_name>`;
- изображения будут ресайзнуты и разложены в `train/valid/test`;
- создадутся `split_map.json` и `hashes.json`;
- опционально будут сгенерированы аугментированные копии.

### 2) Обучение
```bash
python training.py
```
- Загружает базовые веса (см. `model_old` в коде).
- Использует GPU, если доступен (`device=0`), иначе CPU.
- Результаты сохраняются в `models/<model_name>_exp.../`.

### 3) Использование модели

#### Вариант А — простой
Редактируй константы в начале `use_model_easy.py`:
```python
WEIGHTS = r"models\cat&dog_exp\weights\best.pt"
INPUT_PATH = r"data\cat&dog\cat"
IMAGE_SIZE = 224
CONF_THRESHOLD = 0.75
```
Запуск:
```bash
python use_model_easy.py
```

#### Вариант B — CLI
```bash
# классификация папки рекурсивно, батч 64, порог 0.9
python use_model_smart.py --model-name "cat&dog" --input "D:/images" --recursive --conf 0.90 --batch 64
```

Поддерживает множество флагов: `--weights`, `--topk`, `--csv`, `--actions`, `--dry-run`.

---

## Конфигурация
Файл `config.json` в `data/<model_name>/` управляет пайплайном:
```jsonc
{
  "model_name": "cat&dog",
  "train_ratio": 0.60,
  "valid_ratio": 0.25,
  "test_ratio": 0.15,
  "image_size": [224, 224],
  "augmentation_factor": 1,
  "enable_noise_artifacts": true,
  "enable_geometric_transforms": true,
  "force_rebuild": false
}
```

---

## Аугментации
- **Шумы и артефакты**: гауссовский шум, ISO-шум, блюр, JPEG-сжатие.
- **Геометрические**: флипы, поворот, аффинные трансформации, искажения.

`augmentation_factor` задаёт количество копий.

---

## Действия и CSV-отчёты
Скрипт `use_model_smart.py` может не только классифицировать, но и выполнять действия с файлами (перемещать, копировать, удалять).

Пример `actions.json`:
```json
{
  "action_map": { "cat": "move", "dog": "ignore" },
  "dest_root": "D:/sorted",
  "unknown_action": "ignore"
}
```

---

## Проверка CUDA
Запусти `utilities/CUDA_Test.py`, чтобы убедиться в правильной установке PyTorch + CUDA.

---

## Устранение неполадок
- Несовпадение Torch/CUDA → проверь версию драйверов и wheel.
- Out of memory → уменьши `image_size`, дай `batch=-1` или используй CPU.
- Файлы не сортируются → проверь `--actions` и `dest_root`.
- Слэши путей Windows/Unix смешаны → используй кавычки вокруг путей.

---

## FAQ
**Привязан ли проект к YOLOv8?**  
Нет, используется общий интерфейс Ultralytics.

**Можно ли использовать свои веса?**  
Да, укажи путь в `training.py` или флаг `--weights`.

---

## Благодарности
- [Ultralytics](https://github.com/ultralytics/ultralytics) за классификационные модели.
- [Albumentations](https://github.com/albumentations-team/albumentations) за аугментации.
