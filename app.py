import cv2
import base64
import uvicorn
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from pydantic import BaseModel
from fastapi import Form, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from configs.configs import load_configs

configs = load_configs()

# Load a model
face_mask_model = YOLO(configs["model_path"])  # pretrained YOLOv8n model

def get_prediction_result(filename):
# Run batched inference on a list of images
    results = face_mask_model.predict(filename, device=configs["devices_id"],\
                                      imgsz=320, conf=configs["confidence_threshold"], \
                                     iou=configs["iou_threshold"])
    results[0].names = {0: "Incorrect Mask", 1: "Correct Mask", 2: "Not Wearing Mask"}
    return results[0].plot()

app = FastAPI()

templates = Jinja2Templates(directory="templates")

class Request_image(BaseModel):
    data: str
    name: str

@app.get("/", response_class=HTMLResponse)
async def main_UI(request: Request):
    return templates.TemplateResponse("homepage.html", {"request": request})

@app.post("/predict")
async def main_UI(data: str = Form(...), name: str = Form(...), ):
    imgsrcstring = data
    imgdata = imgsrcstring.split(',')[1]
    decoded = base64.b64decode(imgdata)
    img = Image.open(BytesIO(decoded))

    np_img = get_prediction_result(img)
    pil_image = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(pil_image)
    ratio = min(900/img.size[0],900/img.size[1])
    img = img.resize((int(img.size[0]*ratio),int(img.size[1]*ratio)))
    
    rawBytes = BytesIO()
    img.save(rawBytes,"JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.getvalue()).decode('ascii')
    uri= "data:%s;base64,%s"%("image/jpeg",img_base64)
    image=uri
    return {'detected':image}

if __name__ == "__main__":
   uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)