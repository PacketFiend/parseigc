#!/usr/bin/python3
import itertools

import geopy.distance
import pandas as pd

from aerofiles.igc import Reader
import datetime
import argparse
import dataclasses
from typing import List, Optional
import math
import statistics
import opensoar
from opensoar import thermals, task
from opensoar.thermals.flight_phases import FlightPhases
from opensoar.task import waypoint
from opensoar.competition.soaringspot import get_info_from_comment_lines
from opensoar.task.trip import Trip
from opensoar.task.waypoint import Waypoint
from opensoar.task.race_task import RaceTask
from opensoar.task.task import Task
from scipy import constants
import random
import numpy
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
    enl: Optional[List[float]] = None

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
                'York Soaring': (43.838098, -80.440351),
                'Pincher Creek': (49.520621, -114.000705)}

def parse_igc(infile):
    with open(infile, 'r') as f:
        parsed_igc_file = Reader().read(f)
        if args.verbose:
            print("-"*100 + "\n|" + " "*38 + "BEFORE DATA PROCESSING" + " "*38 + "|\n" + "-"*100)
        # Pick ten fixes at random. If all ten have a seconds value of zero, chances are the seconds value is missing.
        # With a fix every five seconds, the chance is 1/1.615*10^11, so not zero, but ¯\_(ツ)_/¯
        # Maybe hacky, maybe brilliant, maybe stupid. I dunno.
        # TODO: The case for one fix per minute still needs to be considered
        timestamps_include_seconds = False
        for i in range(1,10):
            if random.choice(parsed_igc_file['fix_records'][1])['time'].second != 0:
                timestamps_include_seconds = True
                break
        if not timestamps_include_seconds:
            parsed_igc_file = interpolate_seconds(parsed_igc_file)

    return parsed_igc_file

def remove_identical_items_from_start_and_and(lst):
    # I shamelessly admit to using openai to solve this problem
    start = 0
    end = len(lst) - 1

    # Find the index of the last item that is not identical to the first item
    for i in range(len(lst)):
        if lst[i]['time'] != lst[0]['time']:
            start = i
            break

    # Find the index of the first item that is not identical to the last item
    for i in range(len(lst)-1, -1, -1):
        if lst[i]['time'] != lst[-1]['time']:
            end = i
            break

    return lst[start:end+1]

def interpolate_seconds(parsed_igc_file):

    normalized_fixes = remove_identical_items_from_start_and_and(parsed_igc_file['fix_records'][1])

    if "utc_date" in parsed_igc_file['header'][1]:
        flight_date = parsed_igc_file['header'][1]['utc_date']
    elif "declaration_date" in parsed_igc_file['task'][1]:
        flight_date = parsed_igc_file['task'][1]['declaration_date']
    else:
        raise KeyError("Unable to determine flight date: neither utc_date is present in the header nor declaration_date in the task.")

    result = []
    time = datetime.datetime.combine(flight_date, normalized_fixes[0]['time'])
    current_minute = time.minute
    last_timestamp = time
    num_timestamps = 0
    current_minute_timestaps_start = 0  # Pointer to start of the current set of timestamps
    all_fixes = parsed_igc_file['fix_records'][1]

    for i, fix in enumerate(normalized_fixes):
        num_timestamps += 1
        time = datetime.datetime.combine(flight_date, fix['time'])
        current_minute = time.minute
        if i+2 > len(normalized_fixes):
            break
        next_minute = normalized_fixes[i+1]['time'].minute
        if next_minute > current_minute:
            j = 0
            new_time = datetime.datetime.combine(flight_date, normalized_fixes[current_minute_timestaps_start]['time'])
            seconds_per_fix = 60/num_timestamps
            while j < num_timestamps-1:
                seconds = j*seconds_per_fix
                newfix = normalized_fixes[i+j].copy()
                newfix['time'] = (time + datetime.timedelta(0,seconds)).time()
                result.append(newfix)
                # print(f"Appending {(new_time + datetime.timedelta(0,seconds))}; j={j}; num_timestamps: {num_timestamps}; current_minute: {current_minute}")
                j += 1

            num_timestamps = 0

        current_minute_timestaps_start = i

    parsed_igc_file['fix_records'][1] = result

    return parsed_igc_file

def add_missing_seconds_to_fixes(parsed_igc_file):
    # DEPRECATED. Wrote a better algorithm.
    num_fixes = 0
    t_minus1 = datetime.time(0)
    num_time_deltas = 0
    for fix in parsed_igc_file['fix_records'][1]:
        num_fixes += 1
        t = fix['time']
        if t > t_minus1:
            num_time_deltas += 1
        t_minus1 = t

    if args.verbose:
        print(f"I count {num_fixes} fixes and {num_time_deltas} time deltas")
    if num_fixes == num_time_deltas:
        need_to_reprocess = False
    else:
        normalized_fixes = remove_identical_items_from_start_and_and(parsed_igc_file['fix_records'][1])
        need_to_reprocess = True

    # Estimate number of fixes per minute.
    if need_to_reprocess:
        num_fixes = 0
        num_time_deltas = 0
        i = 0
        for fix in normalized_fixes:
            num_fixes += 1
            t = fix['time']
            if 0 < i < len(normalized_fixes)-1:
                time_plus1 = normalized_fixes[i+1]['time']
                time = normalized_fixes[i]['time']
                if time_plus1 > time:
                    num_time_deltas += 1
            i += 1
    num_fixes = len(normalized_fixes) # Correct the length var, now that we've removed a few fixes

    # Fill in the missing seconds with our previous estimate. Unfortunately, we can't do this
    # without first iterating through the whole file to estimate fixes per minute.
    if need_to_reprocess:
        estimated_fixes_per_minute = round(num_fixes/num_time_deltas)   # NB: This is an **estimate**
        fix_time_interval = int(60 / estimated_fixes_per_minute)
        if args.verbose:
            print(f"This is a fucky recorder. I estimate it takes a fix every {fix_time_interval} seconds.")
        num_fixes = 0
        num_time_deltas = 0
        seconds = 0
        normalized_fixes_with_seconds = []
        for i, fix in enumerate(normalized_fixes.copy()):
            t = fix['time']
            if 0 < i < len(normalized_fixes)-1: # Ignore the first and last fix?
                t_plus1 = normalized_fixes[i+1]['time']
                t_minus1 = normalized_fixes[i-1]['time']
                time = normalized_fixes[i]['time']
                if t_minus1 < t_plus1:
                    num_time_deltas += 1
                    seconds = 0
                # Discard any fixes that would have time.seconds greater than 59.
                # This may lead to some fuckery when calculating speed, but ¯\_(ツ)_/¯
                if seconds > 59:
                    continue
                normalized_fixes_with_seconds.append(normalized_fixes[i])
                normalized_fixes_with_seconds[num_fixes]['time'] = time.replace(second=seconds)
                num_fixes += 1
                seconds += fix_time_interval
    parsed_igc_file['fix_records'][1] = normalized_fixes_with_seconds
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
    # TODO: See if we can get metres/second to feet/minute conversion in the scipy.constants library
    """Gets a conversion factor from metres per second (or simple metres) to whatever your heart desires"""
    # I bet there's a conversion library that would be better for this, but wtf, I can't find one
    if unit in ["kts", "knots"]:
        #return 1.9438444924406
        return constants.knot
    elif unit in ["mph", "miles/h", "miles/hour"]:
        #return 2.2369362920544
        return constants.mph
    elif unit in ["kmh", "km/h", "kph"]:
        #return 3.6
        return constants.kmh
    elif unit in ["fpm", "f/m"]:
        # return 196.85039370078738
        return (constants.foot/constants.minute)
    elif unit in ["feet"]:
        return constants.foot
        #return 3.280839895013123
    elif unit in ["miles", ",mile"]:
        return constants.mile
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

    # Style for thermalling phases (lime green)
    styles['thermalling'] = simplekml.Style()
    styles['thermalling'].linestyle.color = "ff20FF00"
    styles['thermalling'].linestyle.width = 2
    styles['thermalling']._id = "thermalling"

    # Style for cruise phases (light blue)
    styles['cruise'] = simplekml.Style()
    styles['cruise'].linestyle.color = "ffDEFF00"
    styles['cruise'].linestyle.width = 2
    styles['cruise']._id = "cruise"

    # Style for engine on phases (red)
    styles['engine_on'] = simplekml.Style()
    styles['engine_on'].linestyle.color = "ff0000ff"
    styles['engine_on'].linestyle.width = 2
    styles['engine_on']._id = "engine_on"

    for style in styles:
        kmldoc.styles.append(styles[style])

def create_fixes(fix_records):

    x_fixes = [fix['lat'] for fix in fix_records]
    y_fixes = [fix['lon'] for fix in fix_records]
    gps_alt_fixes = [fix['gps_alt'] for fix in fix_records]
    pressure_alt_fixes = [fix['pressure_alt'] for fix in fix_records]
    if statistics.stdev(pressure_alt_fixes) > 0: # We have pressure altitude information
        meta_data['pressure_alt'] = 1
    else: # We do not
        meta_data['pressure_alt'] = 0
    if meta_data['pressure_alt']:
        # Altitude is calculated using both pressure and gps altitude, taking the average
        data.alt = [(gps_alt + pressure_alt)/2 for gps_alt, pressure_alt in zip(gps_alt_fixes, pressure_alt_fixes)]
    else:    # We have *no* pressure altitude information
        data.alt = gps_alt_fixes

    data.lat = x_fixes
    data.lon = y_fixes

    return data

def process_data(parsed_igc_file, units):

    data = FlightData()
    meta_data = {}
    # I have no idea why there's an empty list as element zero...
    if "utc_date" in parsed_igc_file['header'][1]:
        flight_date = parsed_igc_file['header'][1]['utc_date']
    elif "declaration_date" in parsed_igc_file['task'][1]:
        flight_date = parsed_igc_file['task'][1]['declaration_date']
    else:
        raise KeyError("Unable to determine flight date: neither utc_date is present in the header nor declaration_date in the task.")

    x_fixes = [fix['lat'] for fix in parsed_igc_file['fix_records'][1]]
    y_fixes = [fix['lon'] for fix in parsed_igc_file['fix_records'][1]]
    gps_alt_fixes = [fix['gps_alt'] for fix in parsed_igc_file['fix_records'][1]]
    pressure_alt_fixes = [fix['pressure_alt'] for fix in parsed_igc_file['fix_records'][1]]

    if statistics.stdev(pressure_alt_fixes) > 0: # We have pressure altitude information
        meta_data['pressure_alt'] = 1
    else: # We do not
        meta_data['pressure_alt'] = 0
    if meta_data['pressure_alt']:
        # Altitude is calculated using both pressure and gps altitude, taking the average
        data.alt = [(gps_alt + pressure_alt)/2 for gps_alt, pressure_alt in zip(gps_alt_fixes, pressure_alt_fixes)]
    else:    # We have *no* pressure altitude information
        data.alt = gps_alt_fixes

    if "ENL" in parsed_igc_file['fix_records'][1]:
        data.enl = [fix['ENL'] for fix in parsed_igc_file['fix_records'][1]]
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
    day = flight_date.day
    month = flight_date.month
    year = flight_date.year
    meta_data['flight_date'] = f"{year}-{month}-{day}"

    units.xfactor = get_conversion_factor(units.x)
    units.yfactor = get_conversion_factor(units.y)
    units.altfactor = get_conversion_factor(units.alt)

    return data, meta_data, units

def avg_alt(parsed_igc_file, meta_data):

    # Adds a "corrected_alt" entry to the fix_records list which averages gps_alt and pressure_alt
    # Assumes that all flight recorders record GPS altitude

    if meta_data['pressure_alt']:   # We have pressure altitude information
        for fix in parsed_igc_file['fix_records'][1]:
            corrected_alt = (fix['pressure_alt'] + fix['gps_alt']) / 2
            fix['corrected_alt'] = int(corrected_alt)
    else:   # We do not have pressure altitude information. Just make it equal to GPS alt.
        for fix in parsed_igc_file['fix_records'][1]:
            fix['corrected_alt'] = fix['gps_alt']

    return parsed_igc_file

def total_distance(flight_data):
    fixes = parsed_igc_file['fix_records'][1]
    total_distance = geopy.distance.Distance(meters=0)

    for i, fix in enumerate(fixes):
        if i+1 >= len(fixes) - 1:
            break
        next_fix = fixes[i+1]
        distance_2d = geopy.distance.geodesic( (fix['lat'], fix['lon']), (next_fix['lat'], next_fix['lon']))
        distance_3d = geopy.distance.Distance(meters=numpy.sqrt( distance_2d.meters**2 + (fix['corrected_alt'] - next_fix['corrected_alt'])**2) )
        total_distance += distance_3d

    return round(total_distance.m)
def write_kml_timeseries(kmldoc, data, speed, units, colormap, n_colors, name="", metric="speed"):

    line_string_folder = kmldoc.newfolder(name=name)
    style = None

    # Assign ranges for colour maps based on mean and standard deviation
    speed_mean = statistics.mean(getattr(data,metric))
    speed_stddev = statistics.stdev(getattr(data,metric))
    cmin = speed_mean - speed_stddev
    cmax = speed_mean + speed_stddev
    postfix = "unknown"
    factor = 1

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
        placemark_name = f'{data.t[i].hour}:{data.t[i].minute}:{data.t[i].second}, {data.alt[i]/units.altfactor:.0f}{units.alt}, {speed[i]/factor:.0f}{postfix}'
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

def write_kml_path(kmldoc, data, extrude=0, style="3", name="Curtain", description=None, folder=None):
    # Keep all the simpleKML library usage in this function, so it's easier to switch to another library later if we need to
    # Put the path in the root, if no folder was supplied
    if folder == None:
        folder = kmldoc
    linestring_coords = [(lon, lat, alt) for lon, lat, alt in zip(data.lon, data.lat, data.alt)]
    line_string = folder.newlinestring(name=name, description=description, coords=linestring_coords)
    line_string.altitudemode = simplekml.AltitudeMode.absolute
    line_string.extrude = extrude
    # Pick out the style we're looking for from the list
    for s in kmldoc.styles:
        if s.id == style:
            path_style = s
            break
    # style_picker = (s for s in kmldoc.styles if s.id == style)
    # style = next(style_picker)
    line_string.style = path_style

def get_taskandtrip(task_data, fix_records):

    trip = None
    # Ignore any waypoints without lat,lon data (some recorders do this for takeoff and landing)
    waypoints = [Waypoint(name = wp['description'],
                          latitude = float(wp['latitude']),
                          longitude = float(wp['longitude']),
                          r_min = None,
                          r_max = 5000,
                          angle_min = 0,
                          angle_max = 180,
                          is_line = False,
                          sector_orientation = "fixed",
                          orientation_angle = 0
                          ) for wp in parsed_igc_file['task'][1]['waypoints'] if wp['latitude'] and wp['longitude']]

    # task, _, _ = get_info_from_comment_lines(parsed_igc_file)
    timezone = -5
    start_opening = None
    start_time = None
    start_time_buffer = 0
    multistart = False

    task = RaceTask(waypoints, timezone, start_opening, start_time_buffer, multistart)
    started = False
    start = None
    for i, fix in enumerate(fix_records):
        if i > 0:
            started, fix, backwards = opensoar.task.task.Task.started(task, fix_records[i-1], fix)
        if started:
            break
    if not started:
        # We never made it to the start point, so use the second waypoint as the start instead.
        if args.verbose:
            print("Unable to determine start point")
        start = waypoints[1]

    trip = Trip(task, parsed_igc_file['fix_records'][1], start)

    return task, trip

def plot_waypoints(kmldoc, task):

    waypoint_folder = kmldoc.newfolder(name="Waypoints")
    last_waypoint = len(task.waypoints) - 1
    i = 0
    for wp in task.waypoints:
        if i < last_waypoint:
            point = waypoint_folder.newpoint(name = wp.name, coords=[(wp.longitude, wp.latitude)])
        else:   # This is the last waypoint. Check if it's equal to the first waypoint.
            if task.waypoints[i] == task.waypoints[0]:
                pass    # Don't plot the same waypoint twice
            else:
                point = waypoint_folder.newpoint(name = wp.name, coords=[(wp.longitude, wp.latitude)])
        i += 1
    # Plot start and finish waypoints, which may be identical and/or the same as the first waypoint
    point = waypoint_folder.newpoint(name="START", coords=[(task.start.longitude, task.start.latitude)])
    point = waypoint_folder.newpoint(name="FINISH", coords=[(task.finish.longitude, task.finish.latitude)])

def plot_fixes(kmldoc, trip):

    fix_folder = kmldoc.newfolder(name="Fixes")
    last_fix = len(trip.fixes) - 1
    i = 1
    for fix in trip.fixes:
        name = f"FIX {i}"
        if "comment" in fix:
            name += "(" + fix['comment'] + ")"
        point = fix_folder.newpoint(name = name, coords=[(fix['lon'], fix['lat'])])
        i += 1

def write_engine_on_path(kmldoc, data, style=None, description=None):

    linestring_coords = [(lon, lat, alt) for lon, lat, alt, enl,t in zip(data.lon, data.lat, data.alt, data.enl, data.t) if enl > 50 and alt > 500]
    folder = kmldoc.newfolder(name="Engine On")
    line_string = folder.newlinestring(name=name, description=description, coords=linestring_coords)
    line_string.altitudemode = simplekml.AltitudeMode.absolute
    line_string.extrude = 0
    # Pick out the style we're looking for from the list
    for s in kmldoc.styles:
        if s.id == style:
            path_style = s
            break
    # style_picker = (s for s in kmldoc.styles if s.id == style)
    # style = next(style_picker)
    line_string.style = path_style

def _debug_print_igc_file_info(parsed_igc_file, task=None, trip=None, meta_data=None):

    # Prints various info from the parsed IGC file for use in analysis and debugging
    print("-" * 100)
    print("|" + " "*36 + "CONTENTS OF PARSED IGC FILE" + " "*35 + "|")
    print("-" * 100)
    for element in parsed_igc_file:
        print(element)

    print("-" * 100)
    print("|" + " "*40 + "TASK FROM IGC FILE" + " "*40 + "|")
    print("-" * 100)
    pprint(parsed_igc_file['task'])

    print("-" * 100)
    print("|" + " "*39 + "HEADER FROM IGC FILE" + " "*39 + "|")
    print("-" * 100)
    pprint(parsed_igc_file['header'])

    if task is not None:
        print("-" * 100)
        print("|" + " "*47 + "TASK" + " "*47 + "|")
        print("-" * 100)
        pprint(vars(task))

    if trip is not None:
        print("-" * 100)
        print("|" + " "*47 + "TRIP" + " "*47 + "|")
        print("-" * 100)
        pprint(vars(trip))

    if meta_data is not None:
        print("-" * 100)
        print("|" + " "*45 + "METADATA" + " "*45 + "|")
        print("-" * 100)
        pprint(meta_data)

if __name__ == '__main__':
    units = Units()
    outfile = "newkml.kml"
    parser = argparse.ArgumentParser(description="IGC to KML converter for flight logs, so they can be viewed with Google Earth.")
    parser.add_argument("--input", "-i", help="Input file name", type=str)
    parser.add_argument("--output", "-o", help="Output file name", type=str)
    parser.add_argument("--xunits", "-x", help="Ground speed units (default kmh)", type=str, choices=["kph", "m/s", "kmh", "mph", "kts", "knots", "miles/h"], default="kmh")
    parser.add_argument("--yunits", "-y", help="Vertical speed units (default mps)", type=str, choices=["m/s", "kmh", "mph", "kts", "fpm", "f/m"], default="m/s")
    parser.add_argument("--altunits", "-a", help="Altitude units (default m)", type=str, choices=["m", "feet", "metres", "metres"], default="m")
    parser.add_argument("--skiptimeseries", "-s", help="Do not generate time series paths (useful to reduce processing time of larger files)", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", help="Verbose debugging output", action="store_true", default=False)
    args = parser.parse_args()
    units.x = args.xunits
    units.y = args.yunits
    units.alt = args.altunits
    args = parser.parse_args()

    parsed_igc_file = parse_igc(args.input)
    # Returns multiple lists of fixes, time, deltas, etc
    data, meta_data, units = process_data(parsed_igc_file, units)
    parsed_igc_file = avg_alt(parsed_igc_file, meta_data)

    kml = simplekml.Kml()
    name = meta_data['launch_site']
    if meta_data['pilot']:
        name += " - " + meta_data['pilot']
    if meta_data['flight_date']:
        name += f": {meta_data['flight_date']}"
    kmldoc = kml.newdocument(name=name)

    write_kml_colormaps(kmldoc)
    if not args.skiptimeseries:
        write_kml_timeseries(kmldoc, data, data.speed, units, 'redtogreen', 9, f"Speed [{units.x}]", metric="speed")
        write_kml_timeseries(kmldoc, data, data.vario, units, 'redtogreen', 9, f"Vario [{units.y}]", metric="vario")
    # Adds an extruded 'curtain' between the flight-path and the ground to easier visualize altitude above ground.
    write_kml_path(kmldoc, data, extrude=1, style="polyline", name="Curtain", description="Right click to show elevation profile")

    # The returned data will have a "waypoints" dict, even if it's empty.
    # If there's no other task data, this will result in a task[1] dict with a single, empty element
    trip = None
    if len(parsed_igc_file['task'][1]) > 1:
        task, trip = get_taskandtrip(parsed_igc_file['task'][1], parsed_igc_file['fix_records'][1])
        plot_waypoints(kmldoc, task)
    else:
        print("No task data")

    if args.verbose:
        _debug_print_igc_file_info(parsed_igc_file, task, trip, meta_data)

    phases = FlightPhases('pysoar', parsed_igc_file['fix_records'][1])
    thermals_folder = kmldoc.newfolder(name="Thermals")
    cruises_folder = kmldoc.newfolder(name="Cruises")
    for phase in phases._phases:
        if not phase.is_cruise:
            fixes = create_fixes(phase.fixes)
            write_kml_path(kmldoc, fixes, extrude=0, style="thermalling", name="Thermals", folder=thermals_folder)
        if phase.is_cruise:
            fixes = create_fixes(phase.fixes)
            write_kml_path(kmldoc, fixes, extrude=0, style="cruise", name="Cruises", folder=cruises_folder)

    if len(parsed_igc_file['task'][1]) > 1 and trip is not None:
        plot_fixes(kmldoc, trip)

    if "ENL" in parsed_igc_file['fix_records'][1]:
        write_engine_on_path(kmldoc, data, style="engine_on")

    model = parsed_igc_file['header'][1]['glider_model']
    registration = parsed_igc_file['header'][1]['glider_registration']
    total_distance = total_distance(data)
    # TODO: use the units struct to print distance travelled better, eg. if xunits is kts then distance should be measured in nautical miles
    kmldoc.description = (f"{model} ({registration}) - {round((total_distance)/get_conversion_factor('miles'))} miles")
    kml.save(args.output)

# EXAMPLE TASK:
# {'_waypoints': [<Waypoint lat=52.44638333333333, lon=6.341116666666666>,
# <Waypoint lat=52.25, lon=6.158333333333333>,
# <Waypoint lat=52.08166666666666, lon=6.446666666666666>,
# <Waypoint lat=52.473333333333336, lon=6.41>,
# <Waypoint lat=52.46888333333333, lon=6.333333333333333>],
# 'distances': [25152.841688455122,
#               27204.052228588032,
#               43653.56790863828,
#               5233.34288151257],
# 'multistart': False,
# 'start_opening': None,
# 'start_time_buffer': 0,
# 'timezone': 2}

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
