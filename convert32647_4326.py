import os
import sys
from os.path import isdir, join, isfile
from os import mkdir
import glob
import dateutil.parser as parserdate
import pyodbc
import math
import shapely
import makevalid

from pycrs.parser import from_epsg_code
import geopandas as gpd
import pandas as pd
import shapely.wkt
from shapely.geometry import Polygon

try:
        from osgeo import ogr, osr, gdal
except:
        sys.exit('ERROR: cannot find GDAL/OGR modules')

import argparse

def read_wkt(polygon_pd_df):
        polygons = []
        for poly_row in polygon_pd_df:
                if poly_row is None:
                        continue
                elif type(poly_row) is str:
                        polygon = ogr.CreateGeometryFromWkt(poly_row)
                        source = osr.SpatialReference()
                        source.ImportFromEPSG(32647)
                        target = osr.SpatialReference()
                        target.ImportFromEPSG(int(4326))
                        transform = osr.CoordinateTransformation(source, target)
                        polygon.Transform(transform)
                        polygons.append(polygon)
                else:
                        continue
        return polygons


#Purpose to change epsg	

df = pd.read_csv('GRID_ALL_BBRD019001.csv', delimiter ='|')
print(df.head(3))
#for i in cnv :
#cnv = s.replace('POLYGON (', '')

def convert_grid_4326(gdf):
		gdf = gdf.to_crs(epsg=4326)
		return gdf

weed_data = df["ContourData"]
weed_poly_fixed = [makevalid.make_geom_valid(shapely.wkt.loads(str(poly)))for poly in weed_data]
weed_data = [poly.wkt for poly in weed_poly_fixed]
weed_poly = read_wkt(weed_data)
df["weed_poly"] = weed_poly
print(df.head(3))