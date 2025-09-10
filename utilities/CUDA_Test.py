import torch

print(f"Версия Python: {torch.__config__.show().splitlines()[0].split()[-1]}")
print(f"Версия torch: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Версия CUDA (из torch): {torch.version.cuda}")
    print(f"Версия cuDNN: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

    num_devices = torch.cuda.device_count()
    print(f"Найдено устройств: {num_devices}")

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        print(f"\n--- Устройство {i} ---")
        print(f"Имя: {props.name}")
        print(f"Вычислительная способность (Compute Capability): {props.major}.{props.minor}")
        print(f"Объем памяти: {props.total_memory / 1024**3:.2f} GB")
        print(f"Мультипроцессоров: {props.multi_processor_count}")