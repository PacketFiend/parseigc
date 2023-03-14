#!/usr/bin/python3

from aerofiles.igc import Reader
import datetime
import argparse
import dataclasses
from typing import List, Optional
import math
import statistics
import pint # unit conversion library

from fastkml import kml
import simplekml

from pprint import pprint
import sys

@dataclasses.dataclass
class FlightData:
    t: Optional[List[datetime.datetime]] = None
    lat: Optional[List[float]] = None
    lon: Optional[List[float]] = None
    alt: Optional[List[float]] = None
    vario: Optional[List[float]] = None
    x: Optional[List[float]] = None
    y: Optional[List[float]] = None
    speed: Optional[List[float]] = None
    timedelta: Optional[List[datetime.timedelta]] = None
    alt_delta: Optional[List[float]] = None

    @property
    def n_samples(self):
        return len(self.t)

class Units:
    x: str
    y: str
    alt: str
    xfactor: Optional[float] = None
    yfactor: Optional[float] = None
    altfactor: Optional[float] = None

EARTH_RADIUS = 6371e3 # radius of the earth in meters
# Add launch-sites in format (latitude, longitude) in decimal notation
LAUNCH_SITES = {"Sonnwendstein": (47.622361, 15.8575),
                'Hohe Wand': (47.829167, 16.041111),
                'Invermere': (50.521301, -116.005644),
                'York Soaring': (43.838098, -80.440351)}

def parse_igc(infile):
    with open(infile, 'r') as f:
        parsed_igc_file = Reader().read(f)
    return parsed_igc_file

def get_nearest_launch_site_name(lat, lon):
    best_name = "Unknown"
    best_distance = 10e3
    for name, (lat_site, lon_site) in LAUNCH_SITES.items():
        d = get_distance(lat, lon, lat_site, lon_site)
        if d < best_distance:
            best_distance = d
            best_name = name
    return best_name

def get_distance(lat1, lon1, lat2, lon2):
    a = math.sin(lat1 * math.pi / 180) * math.sin(lat2 * math.pi / 180)
    b = math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.cos((lon2 - lon1) * math.pi / 180)
    return math.acos(a+b) * EARTH_RADIUS

def get_conversion_factor(unit):
    """Gets a conversion factor from metres per second (or simple metres) to whatever your heart desires"""
    # I bet there's a conversion library that would be better for this, but wtf, I can't find one
    if unit in ["kts", "knots"]:
        return 1.9438444924406
    elif unit in ["mph", "miles/h", "miles/hour"]:
        return 2.2369362920544
    elif unit in ["kmh", "km/h", "kph"]:
        return 3.6
    elif unit in ["fpm", "f/m"]:
        return 196.85039370078738
    elif unit in ["feet"]:
        return 3.280839895013123
    elif unit in ["m/s", "mps", "m", "metres", "meters"]:
        return 1.0
    else:
        raise ValueError(f"Unknown unit for speed: {unit}")


def write_kml_colormaps(kmldoc):
    styles = {}

    # Write colour maps for line styles, for speed and vario visualization
    color_maps = dict(redtogreen=["ff2600a5", "ff2e40de", "ff528ef9", "ff81d4fe", "ffbefffe", "ff82e9cb", "ff66ca84", "ff54a02a", "ff376800"])
    for color_name, color_value in color_maps.items():
        for i, c in enumerate(color_value):
            styles[color_name+str(i)] = simplekml.Style()
            styles[color_name+str(i)].linestyle.color = c
            styles[color_name+str(i)].linestyle.width = 2
            styles[color_name+str(i)]._id = color_name+str(i)

    # Write colour map for extruded "curtain", to visualize altitude
    styles['polyline'] = simplekml.Style()
    styles['polyline'].linestyle.color = "00ff0000"
    styles['polyline'].linestyle.width = 1
    styles['polyline'].polystyle.color = "7fffffff"
    styles['polyline']._id = "polyline"
    for style in styles:
        kmldoc.styles.append(styles[style])

def process_data(parsed_igc_file, units):

    data = FlightData()
    meta_data = {}
    # I have no idea why there's an empty list as element zero...
    #flight_date = parsed_igc_file['task'][1]['flight_date']
    flight_date = parsed_igc_file['header'][1]['utc_date']
    x_fixes = [fix['lat'] for fix in parsed_igc_file['fix_records'][1]]
    y_fixes = [fix['lon'] for fix in parsed_igc_file['fix_records'][1]]
    gps_alt_fixes = [fix['gps_alt'] for fix in parsed_igc_file['fix_records'][1]]
    pressure_alt_fixes = [fix['pressure_alt'] for fix in parsed_igc_file['fix_records'][1]]
    # Altitude is calculated using both pressure and gps altitude, taking the average
    data.alt = [(gps_alt + pressure_alt)/2 for gps_alt, pressure_alt in zip(gps_alt_fixes, pressure_alt_fixes)]
    data.x = [EARTH_RADIUS * math.cos(lat * math.pi / 180) * math.cos(lon * math.pi / 180) for lat, lon in zip(x_fixes, y_fixes)]
    data.y = [EARTH_RADIUS * math.cos(lat * math.pi / 180) * math.sin(lon * math.pi / 180) for lat, lon in zip(x_fixes, y_fixes)]

    # Add the date to all the datetime.time objects and turn them into datetime.datetime. It's not possible to calculate a time delta
    # between datetime.time objects - only datetime.datetime objects
    data.t = [datetime.datetime.combine(flight_date,t['time']) for t in parsed_igc_file['fix_records'][1]]
    data.timedelta = [(data.t[i+1] - data.t[i]).seconds for i in range(data.n_samples - 1)]
    data.distance_delta = [math.sqrt((data.x[i + 1] - data.x[i]) ** 2 + (data.y[i + 1] - data.y[i]) ** 2) for i in
                      range(data.n_samples - 1)] + [0.0]

    # Some recording devices take multiple fixes per minute, but only record timestamps accurate to the minute.
    # Adding one millisecond avoids division by zero exceptions, but fubars the speed calculations.
    # A better method would be to calculate the frequency of fixes in these cases. This is an ugly hack.
    data.speed = [dx / (dt+0.001) for dx, dt in zip(data.distance_delta, data.timedelta)]
    data.alt_delta = [data.alt[i + 1] - data.alt[i] for i in range(data.n_samples - 1)] + [0.0]
    data.vario = [dy / (dt+0.001) for dy, dt in zip(data.alt_delta, data.timedelta)]
    data.lat = x_fixes
    data.lon = y_fixes
    meta_data['speed_max'] = max(data.speed)
    meta_data['speed_min'] = min(data.speed)
    meta_data['speed_stddev'] = statistics.stdev(data.speed)
    meta_data['speed_mean'] = statistics.mean(data.speed)
    meta_data['vario_max'] = max(data.vario)
    meta_data['vario_min'] = min(data.vario)
    meta_data['vario_stddev'] = statistics.stdev(data.vario)
    meta_data['vario_mean'] = statistics.mean(data.vario)
    meta_data['launch_site'] = get_nearest_launch_site_name(parsed_igc_file['fix_records'][1][0]['lat'], parsed_igc_file['fix_records'][1][0]['lon'])
    meta_data['pilot'] = parsed_igc_file['header'][1]['pilot']
    day = parsed_igc_file['header'][1]['utc_date'].day
    month = parsed_igc_file['header'][1]['utc_date'].month
    year =parsed_igc_file['header'][1]['utc_date'].year
    meta_data['flight_date'] = f"{year}-{month}-{day}"

    units.xfactor = get_conversion_factor(units.x)
    units.yfactor = get_conversion_factor(units.y)
    units.altfactor = get_conversion_factor(units.alt)

    return data, meta_data, units

def write_kml_timeseries(kmldoc, data, speed, units, colormap, n_colors, name="", metric="speed"):

    line_string_folder = kmldoc.newfolder(name=name)
    style = None

    # Assign ranges for colour maps based on mean and standard deviation
    speed_mean = statistics.mean(getattr(data,metric))
    speed_stddev = statistics.stdev(getattr(data,metric))
    cmin = speed_mean - speed_stddev
    cmax = speed_mean + speed_stddev

    # Figure out the factor we need to multiply by
    if metric == "speed":
        factor = units.xfactor
        postfix = units.x
    elif metric == "vario":
        factor = units.yfactor
        postfix = units.y

    for i in range(data.n_samples - 1):
        color = int(n_colors * (speed[i] - cmin) / (cmax - cmin) + 0.5)
        if color >= n_colors:
            color = n_colors - 1
        elif color < 0:
            color = 0
        placemark_name = f'{data.t[i].hour}:{data.t[i].minute}:{data.t[i].second}, {data.alt[i]*units.altfactor:.0f}{units.alt}, {speed[i]*factor:.0f}{postfix}'
        line_string = line_string_folder.newlinestring(name=placemark_name, coords=(
            (data.lon[i], data.lat[i], data.alt[i]),
            (data.lon[i+1], data.lat[i+1], data.alt[i+1])
        ))
        line_string.altitudemode = simplekml.AltitudeMode.absolute
        for s in kmldoc.styles:
            if s.id == colormap+str(color):
                style = s
                break
        line_string.style.linestyle.color = style._kml['LineStyle_']._kml['color']
        line_string.style.linestyle.width = 2
    return

def write_kml_path(kmldoc, data, extrude=0, polycolour="00000000", style="3", name="Curtain", description=None):
    # Keep all the simpleKML library usage in this function, so it's easier to switch to another library later if we need to
    #kmldoc.document_name = "New Document"
    linestring_coords = [(lon, lat, alt) for lon, lat, alt in zip(data.lon, data.lat, data.alt)]
    line_string = kmldoc.newlinestring(name=name, description=description, coords=linestring_coords)
    line_string.altitudemode = simplekml.AltitudeMode.absolute
    line_string.extrude = extrude
    # Pick out the style we're looking for from the list
    style_picker = (s for s in kmldoc.styles if s.id == style)
    style = next(style_picker)
    line_string.style = style

if __name__ == '__main__':
    units = Units()
    outfile = "newkml.kml"
    parser = argparse.ArgumentParser(description="IGC to KML converter for flight logs, so they can be viewed with Google Earth.")
    parser.add_argument("input", nargs="+", help="Input file name(s)")
    parser.add_argument("--xunits", "-x", help="Ground speed units (default kmh)", type=str, choices=["kph", "m/s", "kmh", "mph", "kts", "knots", "miles/h"], default="kmh")
    parser.add_argument("--yunits", "-y", help="Vertical speed units (default mps)", type=str, choices=["m/s", "kmh", "mph", "kts", "fpm", "f/m"], default="m/s")
    parser.add_argument("--altunits", "-a", help="Altitude units (default m)", type=str, choices=["m", "feet", "metres", "metres"], default="m")
    args = parser.parse_args()
    units.x = args.xunits
    units.y = args.yunits
    units.alt = args.altunits
    args = parser.parse_args()

    for input_fname in args.input:
        parsed_igc_file = parse_igc(input_fname)
        # Returns multiple lists of fixes, time, deltas, etc
        data, meta_data, units = process_data(parsed_igc_file, units)

    kml = simplekml.Kml()
    name = meta_data['launch_site']
    if meta_data['pilot']:
        name += " - " + meta_data['pilot']
    if meta_data['flight_date']:
        name += f": {meta_data['flight_date']}"
    kmldoc = kml.newdocument(name=name)
    #kmldoc.name = name
    print(kmldoc.name)
    write_kml_colormaps(kmldoc)

    write_kml_timeseries(kmldoc, data, data.speed, units, 'redtogreen', 9, f"Speed [{units.x}]", metric="speed")
    write_kml_timeseries(kmldoc, data, data.vario, units, 'redtogreen', 9, f"Vario [{units.y}]", metric="vario")

    # Adds an extruded 'curtain' between the flight-path and the ground to easier visualize altitude above ground.
    write_kml_path(kmldoc, data, extrude=1, polycolour="7fffffff", style="polyline", name="Curtain", description="Right click to show elevation profile")
    print("Saving...")
    kml.save(outfile)

# logger_id
# fix_records
# task
# dgps_records
# event_records
# satellite_records
# security_records
# header
# fix_record_extensions
# k_record_extensions
# k_records
# comment_records
