from pydantic import BaseModel
from typing import List, Optional

class Image(BaseModel):
    id: int
    title: str
    url: str
    description: str

class ImagePalette(BaseModel):
    id: int
    title: str
    url: str
    description: str
    palette: List[List[int]]

class ImageList(BaseModel):
    page: int
    results: int
    total: int
    total_pages: int
    images: List[Image]
