from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Union
from models import Image, ImageList, ImagePalette
import json
from urllib.parse import urljoin
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

origins = [
    "http://localhost:3000",  # URL de tu aplicación React
    "http://localhost:8000",  # URL de tu servidor FastAPI
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Lista de orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los headers
)

image_folder = Path(__file__).parent / "images"

with open("data.json", "r") as f:
    images_data = json.loads(f.read())

images_data_list = list(images_data.values())

@app.get("/", response_model=ImageList)
async def list_images(request: Request, page: int = 1, limit: int = 6, filter = None, method = None, distance = 14):
    # TODO: Implement the filtering engine
    
    server_url = urljoin(str(request.url), '/')
    start = (page - 1) * limit
    end = start + limit

    if method is None:
        method = "kmeans"

    # URL encode the method name
    method = method.replace(" ", "+")

    for image in images_data_list:
        image["palette"] = image["palettes_by_method"][method]
    
    if filter is None:
        filtered_images = images_data_list
    else:
        filter_color = (int(channel_level) for channel_level in filter.split(","))
        filtered_images = filter_by_cube(filter_color, images_data_list, distance_from_center = int(distance))
        
    total_images = len(filtered_images)
    total_pages = total_images // limit + (1 if total_images % limit > 0 else 0)
    
    images = []
    for image in filtered_images[start:end]:
        image["url"] = f"{server_url}image/{image['filename']}"
        images.append(image)
    
    return {
        "page": page,
        "results": len(filtered_images[start:end]),
        "total": total_images,
        "total_pages": total_pages,
        "images": [Image(**image) for image in images]
    }


@app.get("/methods")
async def get_methods():
    image_info = images_data_list[0]
    return list(image_info["palettes_by_method"].keys())


@app.get("/image/{param}", response_model=None)
async def get_image(request: Request, param: Union[int, str], method = None):
    try:
        param = int(param)
    except:
        pass
    
    if isinstance(param, int):
        server_url = urljoin(str(request.url), '/')
        image_id = str(param)
        if image_id in images_data:
            image = images_data[image_id]
            if method is not None:
                # URL encode the method name
                method = method.replace(" ", "+")
                image["palette"] = image["palettes_by_method"][method]
            else:
                image["palette"] = image["palettes_by_method"]["kmeans"]
            image["url"] = f"{server_url}image/{image['filename']}"
            return ImagePalette(**image)
        raise HTTPException(status_code=404, detail="Image ID not found")
    else:
        # Lógica para cuando 'param' es una cadena
        filename = param
        image_path = image_folder / filename
        if not image_path.is_file():
            raise HTTPException(status_code=404, detail="Filename not found")
        return FileResponse(image_path)
    
class Cube:
    def __init__(self, center, distance):
        self.center_r, self.center_g, self.center_b = center
        self.distance = distance
        self.r_range = range(self.center_r-distance, self.center_r+distance)
        self.g_range = range(self.center_g-distance, self.center_g+distance)
        self.b_range = range(self.center_b-distance, self.center_b+distance)
    
    def contains(self, color):
        r,g,b = color
        if r in self.r_range and g in self.g_range and b in self.b_range:
            return True
        return False
        

def filter_by_cube(color: np.ndarray, images, distance_from_center=14, levels=256):
    filtering_cube = Cube(color, distance_from_center)
    filtered_images = []
    for image in images:
        palette = image["palette"]
        for priority, color in enumerate(palette):
            if filtering_cube.contains(color):
                filtered_images.append((image, priority))
                break
    filtered_images = sorted(filtered_images, key = lambda x: x[1])
    filtered_images = [element[0] for element in filtered_images]
    return filtered_images
