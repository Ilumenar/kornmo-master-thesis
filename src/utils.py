import math
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from fiona import collection
from fiona.crs import from_epsg
from shapely.geometry import mapping

#Groups a dataframe, gdf, by column, counts the values and plots it as a bar plot. 
def plot_bar(gdf, column):
    plt.figure()
    gdf.groupby(column)[column].count().plot(kind='bar')
    plt.show()


true_color = lambda x: [x[3], x[2], x[1]]

#Extracts the r, g and b-values from the 12-band satellite images and normalizes the values.
def to_rgb(img, map_func=true_color, normalize=True):
    shape = img.shape
    newImg = []    
    for row in range(0, shape[0]):
        newRow = []
        for col in range(0, shape[1]):
            newRow.append(map_func(img[row][col]))
        newImg.append(newRow)
    if normalize:
        normImg = normalize_img(newImg, 1)
        return normImg
    else:
        return newImg

def plot_image(image):
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    #image = image/np.amax(image)
    #image = np.clip(image, 0, 1)
    ax.imshow(image)

    ax.set_xticks([])
    ax.set_yticks([])

def normalize_img(img, new_max):
    min = np.min(img)
    max = np.max(img)
    new_img = []
    for i, row in enumerate(img):
        new_row = []
        for j, pixel in enumerate(row):
            new_rgb = []
            for color in pixel:
                #zi = (xi – min(x)) / (max(x) – min(x)) * 1,000
                new_color = (color - min)/(max - min) * new_max
                new_rgb.append(new_color)
            new_row.append(new_rgb)
        new_img.append(new_row)
    return np.array(new_img)

#
# The following methods for calculating a bounding box in WGS84 is taken from:
# https://stackoverflow.com/a/238558
#


def deg2rad(degrees):
    return math.pi*degrees/180.0


# radians to degrees
def rad2deg(radians):
    return 180.0*radians/math.pi


# Semi-axes of WGS-84 geoidal reference
WGS84_a = 6378137.0  # Major semiaxis [m]
WGS84_b = 6356752.3  # Minor semiaxis [m]


# Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
def WGS84EarthRadius(lat):
    # http://en.wikipedia.org/wiki/Earth_radius
    
    An = WGS84_a*WGS84_a * math.cos(lat)
    Bn = WGS84_b*WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    return math.sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) )


# Bounding box surrounding the point at given coordinates,
# assuming local approximation of Earth surface as a sphere
# of radius given by WGS84
def boundingBox(latitudeInDegrees, longitudeInDegrees, halfSideInKm):
    lat = deg2rad(latitudeInDegrees)
    lng = deg2rad(longitudeInDegrees)
    halfSide = 1000*halfSideInKm

    # Radius of Earth at given latitude
    radius = WGS84EarthRadius(lat)
    # Radius of the parallel at given latitude
    pradius = radius*math.cos(lat)

    lat_min = lat - halfSide/radius
    lat_max = lat + halfSide/radius
    lng_min = lng - halfSide/pradius
    lng_max = lng + halfSide/pradius

    return (rad2deg(lng_min), rad2deg(lat_min),rad2deg(lng_max), rad2deg(lat_max))


#Writes the inputted polygons to the file called filename.
def write_polygons_to_shp(polygons, filename):
    schema = {'geometry': 'Polygon'}
    with collection('../shapefiles/' + filename + '.shp', "w", crs=from_epsg(4326), driver="ESRI Shapefile", schema=schema) as output:
        for polygon in polygons:
            output.write({'geometry': mapping(polygon)})
    output.close()

#Plots the inputted polygons as pyplot
def plot_polygons(polygons):
    for p in polygons:
        plt.plot(*p.exterior.xy)

#Converts a png-image, named by the input 'orgnr' and converts it to a geotiff file that can be shown in qgis.
#The positional data is fetched from a bounding_box created by the BoundingBox method. 
def png_to_geotiff(org_nr, bounding_box):
    dataset = rasterio.open(org_nr + '.png', 'r')
    bands = [1, 2, 3]
    data = dataset.read(bands)
    transform = rasterio.transform.from_bounds(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], data.shape[1], data.shape[2])
    crs = {'init': 'epsg:4326'}

    with rasterio.open(org_nr + '.tif', 'w', driver='GTiff',
                    width=data.shape[1], height=data.shape[2],
                    count=3, dtype=data.dtype,
                    transform=transform, crs=crs) as dst:
        dst.write(data, indexes=bands)

