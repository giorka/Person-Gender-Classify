from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from torch import Tensor
from ultralytics import YOLO

import settings

model: YOLO = YOLO(settings.MODEL_PATH)
folders: iter = Path(f'datasets/{settings.DATASET_NAME}/train').iterdir()
folder_names: list[str] = [folder.name for folder in folders]

app: FastAPI = FastAPI()


@app.post('/')
async def on_file(file: UploadFile = File(...)) -> dict:
    file_name: str = 'image-{uuid}.jpg'.format(uuid=uuid4())  # TODO: reduce size
    content: bytes = await file.read()

    if not content:
        raise HTTPException(status_code=500, detail='Can Not Process The Image.')

    with open(file=file_name, mode='wb') as file:
        file.write(content)

    result: list = model(file_name)

    Path(file_name).unlink()

    probs: list[Tensor] = result[0].probs.data

    return {name: percent for name, percent in zip(folder_names, [round(float(prob) * 100, 3) for prob in probs])}
