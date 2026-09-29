import os
import requests

from bs4 import BeautifulSoup
from tqdm import tqdm

def Mediafire_Download(url, output=None, filename=None):
    if not filename: filename = url.split('/')[-2]
    if not output: output = os.path.dirname(os.path.realpath(__file__))
    output_file = os.path.join(output, filename)

    sess = requests.session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6)"})

    try:
        with requests.get(BeautifulSoup(sess.get(url).content, "html.parser").find(id="downloadButton").get("href"), stream=True) as r:
            r.raise_for_status()
            total_length = int(r.headers.get('content-length'))

            with open(output_file, "wb") as f, tqdm(
                desc=filename,
                total=total_length,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        return output_file
    except Exception as e:
        raise RuntimeError(e)
