import rasterio
import numpy as np
from rasterio.mask import mask
import fiona


class Tiff:
    def __init__(self) -> None:
        pass
        
    @staticmethod
    def process_raster(raster_path, mask_path) -> np.array:
        if 'WV03' in raster_path:
            with rasterio.open(raster_path) as src:
                width = src.width
                height = src.height
                meta = src.meta
                if mask_path:
                    # Mask out land
                    with fiona.open(mask_path, "r") as shapefile:
                        sea_ice_shapes = [feature["geometry"] for feature in shapefile]
                        img, _ = mask(src, shapes=sea_ice_shapes)
                    img = img[[2, 3, 5]].transpose(1, 2, 0)
                else:
                    if "M1BS" in raster_path:
                        img = src.read([2, 3, 5]).transpose(1, 2, 0)
                    else:
                        img = src.read(1).reshape(height, width, 1)[::4, ::4, :]
                        height = img.shape[0]
                        width = img.shape[1]

                
        else:
            raise NotImplementedError(f'sensor not supported for {raster_path}')

        return img, width, height, meta