from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Union
from models import Image, ImageList, ImagePalette
import json
from urllib.parse import urljoin

app = FastAPI()

image_folder = Path(__file__).parent / "images"

with open("data.json", "r") as f:
    images_data = json.loads(f.read())

images_data_list = list(images_data.values())

@app.get("/", response_model=ImageList)
async def list_images(request: Request, page: int = 1, limit: int = 6):
    # TODO: Implement the filtering engine
    
    server_url = urljoin(str(request.url), '/')
    start = (page - 1) * limit
    end = start + limit
    total_images = len(images_data_list)
    total_pages = total_images // limit + (1 if total_images % limit > 0 else 0)

    images = []
    for image in images_data_list[start:end]:
        image["url"] = f"{server_url}{image['url']}"
        images.append(image)
    
    return {
        "page": page,
        "results": len(images_data_list[start:end]),
        "total": total_images,
        "total_pages": total_pages,
        "images": [Image(**image) for image in images]
    }


@app.get("/image/{param}", response_model=None)
async def get_image(request: Request, param: Union[int, str]):
    try:
        param = int(param)
    except:
        pass
    
    if isinstance(param, int):
        server_url = urljoin(str(request.url), '/')
        image_id = str(param)
        if image_id in images_data:
            image = images_data[image_id]
            image["url"] = f"{server_url}{image['url']}"
            return ImagePalette(**image)
        raise HTTPException(status_code=404, detail="Image ID not found")
    else:
        # Lógica para cuando 'param' es una cadena
        filename = param
        image_path = image_folder / filename
        if not image_path.is_file():
            raise HTTPException(status_code=404, detail="Filename not found")
        return FileResponse(image_path)
    
