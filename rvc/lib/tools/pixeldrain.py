import os
import requests
from tqdm import tqdm

def pixeldrain(url, output_dir):
    try:
        response = requests.get(
            f"https://pixeldrain.com/api/file/{url.split('pixeldrain.com/u/')[1]}",
            stream=True
        )

        if response.status_code == 200:
            file_path = os.path.join(
                output_dir,
                response.headers.get("Content-Disposition").split("filename=")[-1].strip('";')
            )

            total_size = int(response.headers.get("Content-Length", 0))

            with open(file_path, "wb") as newfile, tqdm(
                desc=os.path.basename(file_path),
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        newfile.write(chunk)
                        bar.update(len(chunk))

            return file_path
        else:
            return None
    except Exception as e:
        raise RuntimeError(e)
