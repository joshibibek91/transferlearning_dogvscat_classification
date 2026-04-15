

from typing import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException


#Testing the models
import tensorflow
import keras
import numpy as np
from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img

app = FastAPI()
model = keras.models.load_model('my_model.keras')

import shutil
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/save-image/")
async def save_image(file: UploadFile = File(...)):
    destination = UPLOAD_DIR / file.filename
    # Save the uploaded file to the destination path
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
        test_image = load_img(destination,target_size = (224,224,3)) #224 224
        test_image = tensorflow.keras.preprocessing.image.img_to_array(test_image)/255
        test_image = np.expand_dims(test_image, 0)
        prediction = model.predict(test_image)
        result = np.argmax(prediction)
    
        
        
        if result == 0 :
            output = "Cat"
        else:
            output = "Dog"
        
        
    return {"info": f"File saved to {destination}", "output": output}