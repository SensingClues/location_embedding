import math
import numpy as np

def deg2num(zoom, lat_deg, lon_deg):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi)
        / 2.0
        * n
    )
    return zoom, xtile, ytile


def number_of_tiles(zoom):
    """Return number of tiles in total for a given zoom level.

    Taken from https://wiki.openstreetmap.org/wiki/Zoom_levels.
    """

    d = {0 	:1,
         1	:4,
         2	:16,
         3	:64,
         4	:256,
         5	:1024,
         6	:4096,
         7	:16384,
         8	:65536,
         9	:262144,
         10	:1048576,
         11	:4194304,
         12	:16777216,
         13	:67108864,
         14	:268435456,
         15	:1073741824,
         16	:4294967296,
         17	:17179869184,
         18	:68719476736,
         19	:274877906944,
         20	:1099511627776}
    return d[zoom]

def tile_id(x:int, y:int, zoom=14):
    """Create a unique tile id for a given zoom level."""
    x, y = int(x), int(y)
    i = np.ceil(np.log10(np.sqrt(number_of_tiles(zoom)))).astype('int')
    id = int(f"{x:0{i}d}{y:0{i}d}")
    return id

