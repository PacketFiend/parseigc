#!/usr/bin/python3

from aerofiles.igc import Reader
import datetime
import argparse
import dataclasses
from typing import List, Optional
import math

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
    launch_site: Optional[str] = None

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

def process_data(parsed_igc_file, units):

    data = FlightData()
    # I have no idea why there's an empty list as element zero...
    flight_date = parsed_igc_file['task'][1]['flight_date']
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
    data.launch_site = get_nearest_launch_site_name(parsed_igc_file['fix_records'][1][0]['lat'], parsed_igc_file['fix_records'][1][0]['lon'])
    data.lat = x_fixes
    data.lon = y_fixes

    return data

def write_kml_timeseries(data, kmldoc, extrude=0, polycolour="00000000", linecolour=simplekml.Color.red):
    # Keep all the simpleKML library usage in this function, so it's easier to switch to another library later if we need to
    kmldoc.document_name = "New Document"
    linestring_coords = [(lon, lat, alt) for lon, lat, alt in zip(data.lon, data.lat, data.alt)]
    line_string = kmldoc.newlinestring(name="Path", description="Description", coords=linestring_coords)
    line_string.altitudemode = simplekml.AltitudeMode.absolute
    line_string.extrude = extrude
    line_string.linestyle.color = linecolour
    line_string.polystyle.color = polycolour

if __name__ == '__main__':
    units = Units()
    parser = argparse.ArgumentParser(description="IGC to KML converter for flight logs, so they can be viewed with Google Earth.")
    parser.add_argument("input", nargs="+", help="Input file name(s)")
    parser.add_argument("--xunits", "-x", help="Ground speed units (default kmh)", type=str, choices=["m/s", "kmh", "mph", "kts", "knots", "miles/h"], default="kmh")
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
        data = process_data(parsed_igc_file, units)

    kmldoc = simplekml.Kml()
    # Adds an extruded 'curtain' between the flight-path and the ground to easier visualize altitude above ground.
    write_kml_timeseries(data, kmldoc, extrude=1, polycolour="7fffffff", linecolour="00ffffff")
    write_kml_timeseries(data, kmldoc, polycolour=simplekml.Color.red, linecolour=simplekml.Color.red)
    kmldoc.save('newkml.kml')

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
