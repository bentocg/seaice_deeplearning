from functools import partial

import rasterio
import geopandas as gpd
from rasterio.windows import Window
from rasterio.features import shapes
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union


import time
import pandas as pd
from multiprocessing import Pool, cpu_count

from itertools import product


def process_tile(tile, input_raster):
    x, y, x_size, y_size, mask = tile

    # get polygons for window
    with rasterio.open(input_raster) as src:

        crs = src.crs
        window = Window(y, x, y_size, x_size)
        transforms = rasterio.windows.transform(window, src.transform)
        image = src.read(1, window=window)  # first band
        if 0 in mask.shape:
            results = []
        else:
            results = (
                shape["coordinates"][0]
                for shape, _ in shapes(
                    image,
                    mask=mask,
                    transform=transforms,
                )
            )

    geometry = []
    for coords in results:
        try:
            pol = Polygon(coords)
            geometry.append(pol)
        except:
            continue

    # merge overlapping polygons and simplify geometry
    if geometry:
        try:
            geometry = list(unary_union(geometry))
        except:
            return gpd.GeoDataFrame()
        gdf_size = gpd.GeoDataFrame(crs=crs, geometry=geometry)
        gdf_size["geometry"] = gdf_size["geometry"].simplify(3)

        # separate border polygons from inside polygons
        border = LineString(
            [
                Point(transforms * (0, 0)),
                Point(transforms * (0, y_size)),
                Point(transforms * (x_size, y_size)),
                Point(transforms * (x_size, 0)),
                Point(transforms * (0, 0)),
            ]
        )
        border = border.buffer(1)
        gdf_size["is_border"] = [pol.intersects(border) for pol in gdf_size.geometry]
        return gdf_size

    else:
        print(f"No polygons for window ({x}-{x + x_size},{y}-{y + y_size})")
        return gpd.GeoDataFrame()


def polygonize_raster(input_raster, final_output, size=1000):
    print(f"Polygonizing output for scene {input_raster}")

    tic_total = time.time()
    with rasterio.open(input_raster) as src:
        raster_height = src.height
        raster_width = src.width
        crs = src.crs
    x = [
        ele[0]
        for ele in product(range(0, raster_height, size), range(0, raster_width, size))
    ]
    y = [
        ele[1]
        for ele in product(range(0, raster_height, size), range(0, raster_width, size))
    ]
    x_size = [min(size, raster_height % max(1, ele)) for ele in x]
    y_size = [min(size, raster_width % max(1, ele)) for ele in y]
    tiles = (
        [
            x[idx],
            y[idx],
            x_size[idx],
            y_size[idx],
            final_output[x[idx] : x[idx] + x_size[idx], y[idx] : y[idx] + y_size[idx]],
        ]
        for idx in range(len(x))
    )

    process = partial(process_tile, input_raster=input_raster)

    pool = Pool(cpu_count() - 1)
    gdf = gpd.GeoDataFrame(pd.concat(pool.map(process, tiles)))
    pool.close()
    pool.join()

    if "geometry" in gdf.columns:
        new_geometry = gdf.loc[gdf.is_border == 1]["geometry"].buffer(1)
        if len(new_geometry) > 1:
            new_geometry = list(unary_union(new_geometry))
            border_pols = gpd.GeoDataFrame(crs=crs)
            border_pols["geometry"] = new_geometry
            border_pols["geometry"] = border_pols["geometry"].buffer(-1)
            gdf.loc[gdf.is_border == 1, "geometry"] = border_pols["geometry"]
    print(
        f"Processed scene {input_raster} to {len(gdf)} polygons with a patch_sizee of {size} in {(time.time() - tic_total):.2f} seconds."
    )

    return gdf
