import requests


url = "https://www.nga.gov/bin/ngaweb/collection-search-result/search.pageSize__90.pageNumber__{page_number}.lastFacet__artobj_downloadable.json?sortOrder=DEFAULT&artobj_downloadable=Image_download_available&_=1701488520985"


#%%

import asyncio
import aiohttp
from tqdm import tqdm

async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    """
    Asynchronously fetch a URL using aiohttp.

    Args:
        session (aiohttp.ClientSession): The session for making HTTP requests.
        url (str): The URL to fetch.

    Returns:
        str: The response text from the URL.
    """
    try:
        async with session.get(url) as response:
            response_json = await response.json()
            download_files = []
            for result in response_json["results"]:
                download_files.append(result["download"])
            return download_files
    except Exception as e:
        print(f"An error occured with url: {url}. {e}")
        return []

async def run_batch(session: aiohttp.ClientSession, urls: list[str]) -> list:
    return await asyncio.gather(*(fetch(session, url) for url in urls))

async def main(urls: list[str], batch_size: int):
    async with aiohttp.ClientSession() as session:
        # Dividir las URLs en lotes
        batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]

        all_responses = []
        for batch in tqdm(batches):
            responses = await run_batch(session, batch)
            all_responses.extend(responses)

    return all_responses

#%%

urls = [url.format(page_number=i) for i in range(1, 638)]
batch_size = 50
# Run the main function
data = asyncio.run(main(urls, batch_size))


#%%
new_data = [url for list_urls in data for url in list_urls]

#%%
import pandas as pd

serie = pd.Series(new_data, name="url")
serie.to_csv("urls_images_national_gallery_art.csv", index=False)
