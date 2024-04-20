from pathlib import Path

from ultralytics import YOLO

import settings

instance: YOLO = YOLO(settings.BASE_MODEL)

if not Path(settings.MODEL_PATH).is_file():
    instance.train(data=settings.DATASET_NAME, epochs=settings.EPOCHS)
