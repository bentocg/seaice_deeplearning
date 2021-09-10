import rasterio
import numpy as np


class Tiff:
    def __init__(self) -> None:
        pass
        
    @staticmethod
    def process_raster(raster_path) -> np.array:
        if 'WV03' in raster_path:
            with rasterio.open(raster_path) as src:
                img = src.read([2, 3, 5]).transpose(1, 2, 0)
                
        else:
            raise NotImplementedError(f'sensor not supported for {raster_path}')
        width = src.width
        height = src.height
        meta = src.meta
        return img, width, height, meta