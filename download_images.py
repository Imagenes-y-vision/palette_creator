import pandas as pd
import asyncio
import aiohttp
import aiofiles
from tqdm import tqdm
import io
from PIL import Image

#%%
urls = list(pd.read_csv("urls_images_national_gallery_art.csv")["url"])


#%%

async def download(session, url, item_id):
    filename = f"data/national_gallery_art/{item_id}.jpg"
    if isinstance(url, str):
        async with session.get(url) as response:
            image_data = await response.read()
            if response.status == 200:
                try:
                    with Image.open(io.BytesIO(image_data)) as img:
                        img_resized = img.resize((512, 512))
                        img_resized.save(filename)
                except UnidentifiedImageError:
                    print(f"No se pudo identificar la imagen en la URL: {url}")
                except:
                    print(f"Otro error ocurrió al tratar de abrir la imagen con Image y BytesIO en la URL: {url}")
            else:
                print(f"Error al descargar la imagen de la URL: {url}, Estado: {response.status}")
                    # async with aiofiles.open(filename, "wb") as f:
                    #     await f.write(await response.read())
    else:
        print(f"Error al obtener la url de la sesión para URL: {url}")
async def run_batch(session: aiohttp.ClientSession, urls: list[str]) -> list:
    return await asyncio.gather(*(download(session, url_image, item_id) for item_id, url_image in urls))

async def main(urls: list[str], batch_size: int):
    async with aiohttp.ClientSession() as session:
        #urls = [(item_id, url) for item_id, url in enumerate(urls)]
        # Dividir las URLs en lotes
        batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]

        for batch in tqdm(batches):
            await run_batch(session, batch)



#%%
#urls = urls[:100]
batch_size = 100
asyncio.run(main(urls, batch_size))
print("All images downloaded")


#%%
import os

current_gallery_art = os.listdir("data/national_gallery_art")
#current_gallery_art = [int(item_id) for item_id.spl]

current_gallery_art = set(int(image.split(".")[0]) for image in current_gallery_art)
urls_not_downloaded = [(idx, url) for idx, url in enumerate(urls) if idx not in current_gallery_art]

#%%
batch_size = 100
asyncio.run(main(urls_not_downloaded, batch_size))
print("All images downloaded")
