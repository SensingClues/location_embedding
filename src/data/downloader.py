import numpy as np
import requests
import shutil
from src.utils.geo import deg2num

def get_tile(lat=[22.946139, 23.327514], lon=[77.040589, 77.5119181], zoom=14, root="data/"):
    """
    
    Parameters
    ----------
    lat: list
        Latitude boundaries in degrees
    lon: list
        Longitude boundaries in degrees
    zoom: int
        zoom level

    Returns
    -------

    """
    _, _x, _y = np.array([deg2num(zoom, _lat, _lon) for _lat, _lon in zip(lat, lon)]).T
    _x.sort()
    _y.sort()
    startx, endx = _x[0], _x[-1]
    starty, endy = _y[0], _y[-1]
    x_vals = np.arange(startx, endx + 1)
    y_vals = np.arange(starty, endy + 1)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    N_tiles = len(x_grid.flatten())
    print("Start downloading tiles")
    for ix, (x, y) in enumerate(zip(x_grid.flatten(), y_grid.flatten())):
        r = requests.post(f"https://a.tile.opentopomap.org/{zoom}/{x}/{y}.png", stream=True)
        path = f"{root}/tiles/opentopomap_{zoom}_{x}_{y}.png"
        if r.status_code == 200:
            with open(path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
        del r

        if ix % 10 == 0:
            print(f"{ix} / {N_tiles}")

if __name__ == '__main__':
    get_tile()