def GeoJSONToFC(shpname,USGSurl):
    from urllib.request import urlopen
    import json
    from shapely.geometry import Point, mapping
    from fiona import collection
    from fiona.crs import from_epsg
    crs = from_epsg(4326)
    schema = {'geometry': 'Point', 'properties': { 'Place': 'str', 'Magnitude': 'str' }}
    with collection(shpname, "w", crs=crs, driver="ESRI Shapefile", schema=schema) as output:
        url = USGSurl
        weburl = urlopen(url)
        if weburl.getcode() == 200:
            data = json.loads(weburl.read())
        for i in data["features"]:
            mag, place = i["properties"]["mag"],i["properties"]["place"]
            x, y = float(i["geometry"]["coordinates"][0]), float(i["geometry"]["coordinates"][1])
            point = Point(x,y)
            output.write({'properties':{'Place': place, 'Magnitude': mag}, 'geometry': mapping(point)})

shpname = "quakes.shp"
url = "http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson"
#GeoJSONToFC(shpname,url)



