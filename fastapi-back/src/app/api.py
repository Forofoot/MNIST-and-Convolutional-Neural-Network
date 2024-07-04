import sys

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np

sys.path.append("fastapi-back/src/model")
from notebook import ConvNet

app = FastAPI()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Grayscale(num_output_channels=1)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet(input_size=(28, 28), n_kernels=6, output_size=10)
model.load_state_dict(torch.load("model/mnist-0.0.1.pt", map_location=device))
model.eval()

@app.post("/api/v1/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    prediction = predict(image)
    return JSONResponse(content={"prediction": prediction})

def predict(image: Image.Image):
    image = image.resize((28, 28)).convert("L")
    image = np.array(image).astype(np.float32)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(image)
        prediction = torch.argmax(logits, dim=1).item()

    return prediction

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)