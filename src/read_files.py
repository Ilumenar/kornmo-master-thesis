import fiona

shape = fiona.open \
    ('../../kornmo-data-files/raw-data/farm-information/farm-properties/bounding-boxes-previous-students'
     '/disponerte_eiendommer_bboxes.shp')

print(shape.schema)

# first feature of the shapefile
first = shape.next()

print(first)
