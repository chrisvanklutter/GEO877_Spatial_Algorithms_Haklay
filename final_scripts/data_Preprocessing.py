#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Basic geometries from GEO877 with some custom Classes for Buffer
from geospatial import *

# Plotting, Arrays, Filepath handling and basic flattenning nested lists
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import chain

# Transformation of OSM data to projected CRS
from pyproj import Transformer

# OSM Part (and some selected apply() function calls)
import geopandas as gpd

# ASTRA Part (to load geodatabase)
import fiona

# Writing to results into pickles for caching between parts
import pickle

# For Preprocessing part - 
plot_steps = False
overwrite_pickles = True


# In[ ]:


# Define class to store attributes with each motorway segment
class MotorwaySegment(Segment):
    """
    MotorwaySegments class to store original data in Basic geometry form.
    In addition to base properties of geospatial.Segments following properties are defined:
    - name (str): Numeric string of custom name within our script. (e.g.: "1")
    - origin_name (str): Identifier/name in original data source (e.g.: "A1")
    - source (str): String corresponding to shortname used to declare origin (e.g.: "osm")
    - crs (str): String to identify Coordinate Reference System used as EPSG string (e.g.:'epsg:2056')
    """
    def __init__(self, start, end, name=str(), origin_name=str(), source=str(), crs=str()):
        # Initialize Segment superclass
        super().__init__(start, end)
        self.name = name
        self.origin_name = origin_name
        self.source = source
        self.crs = crs
    
    # Overwrite representation to include name of Segment
    def __repr__(self):
        return f'MotorwaySegment of "{self.name}" with start {self.start} and end {self.end}.' 
    
    def plot_seg(self, ax = None, color = 'blue', linewidth = 1, label = 'segments'):
        """
        Plots a segment as line within matplotlib.pyplot figure.
        """
        # List of x- and y- coordinates
        xs = [self.start.x, self.end.x]
        ys = [self.start.y, self.end.y]
        # Add to custom ax
        if ax:
            ax.plot(xs, ys, color = color, linewidth = linewidth, label=label)
        # Use default ax
        else:
            plt.plot(xs, ys, color = color, linewidth = linewidth, label=label)



# In[ ]:


class MotorwayLine():
    """
    MotorwayLine class - LineString-like object to store original data in Basic geometry form for whole highways.
    - segments (list): List of MotorwaySegment class objects
    - name (str): Numeric string of custom name within our script. (e.g.: "1"). Assumes all MotorwaySegments passed have the same name.
    - points (list): List of geospatial.Point objects
    - points_x (list): List of x-coordinates
    - points_y (list): List of y-coordinates
    """
    #initialise
    def __init__(self, data=None, xcol=None, ycol=None): #data = list of segments from the same street
        self.segments = []
        self.name = data[0].name
        self.points = []
        self.points_x = []
        self.points_y = []
        
        for d in data:
            #add segment
            self.segments.append(d)
            #add points - there will be duplicates /!\
            self.points.append(d.start)
            self.points.append(d.end)
            self.points_x.append(d.start.x)
            self.points_x.append(d.end.x)
            self.points_y.append(d.start.y)
            self.points_y.append(d.end.y)  
    
    def __len__(self):
        return len(self.segments)
    
        # Test for equality between Segments - we treat segments going in opposite directions as equal here
    def __eq__(self, other): 
        if (self.points_x == other.points_x or self.points_x[::-1] == other.points_x) and (self.points_y == other.points_y or self.points_y[::-1] == other.points_y) :
            return True
        else:
            return False
            
    # We need this method so that the class will behave sensibly in sets and dictionaries
    def __hash__(self):
        return hash((self.segments, self.points_x, self.points_y)) 
    
    def removeDuplicates(self, print_change=False):
        # Note: we could use inheritance within MotorwaySegments to remove duplicates
        # Copy of geospatial.Segment
        oldn = len(self.points)
        self.points = list(dict.fromkeys(self.points)) # Get rid of the duplicates
        n = len(self.points)
        if print_change:
            print(f'The old line had {oldn} points, now we only have {n}.')
        return self
        
    def bbox(self):
        # Copy of geospatial
        x = [i.x for i in self.points]   # extract all x coords as a list
        y = [i.y for i in self.points]   # extract all y coords as a list

        # determine corners, calculate centre and area
        self.ll = Point(min(x), min(y))    # lower-left corner (min x, min y)
        self.ur = Point(max(x), max(y))    # upper-right corner (max x, max y)
        self.ctr = Point((max(x)-min(x))/2, (max(y)-min(y))/2)   # centre of box
        self.area = (abs(max(x)-min(x)))*abs((max(y)-min(y)))    # area of box
        print(f'Bounding box with lower-left {self.ll} and upper-right {self.ur}' )
        return self.ll, self.ur
    
    def plot_bbox(self):
        """
        Plots bounding box of MotorwayLine in question in current axis.
        """
        ll, ur = self.bbox()
        plt.plot([ll.x, ll.x, ur.x, ur.x, ll.x], [ll.y, ur.y, ur.y, ll.y, ll.y], linestyle='dashed')

    def plot_seg(self, ax = None, color = 'blue', linewidth = 1, label = 'segments'):
        """
        Plots a segment as line within matplotlib.pyplot figure.
        """
        for seg in self.segments:
            seg.plot_seg(ax = ax, color = color, linewidth = linewidth, label = label)

        
     # Overwrite representation
    def __repr__(self):
        return f'MotorwayLine "{self.name}" has "{len(self.segments)}" segments, and "{len(self.points)}" points.'    


# In[ ]:


def name_mapping(in_name):
    """
    Single use function to map names of the OSM data to custom names e.g.: A1;A4 -> 1; A1 -> 1
    """
    #special cases, checked on QGIS:
    name_map = {"32": 1,
            "A1;A4": 1,
            "A1;A2": 1,
            "A1;A3": 1,
            "A1;A6": 1,
            "A1;A9": 1,
            "A9;A 2": 2,
            "A2;A3": 2,
            "A3;A4": 3,
            "53": 15,
            "A20;20": 20
           }
    if in_name in name_map.keys():
        return name_map[in_name]

    #correct cases:
    if ";" not in in_name:
        return int(''.join(filter(str.isdigit, in_name)))
    
    #catch new roads names
    else:
        print(in_name)
        print("Name not treated")


# In[ ]:


def linestring_to_motorwaysegment_list(ls_seg):
    name = ls_seg["name"]
    name_origin = ls_seg["ref"]
    source = ls_seg["source"]
    crs = ls_seg["crs"]
    point_list = tuple(Point(x, y) for x, y in zip(*ls_seg.geometry.coords.xy))
    if len(point_list) <= 1:
        print("Cannot build MotorwaySegment from line with less than 2 points!")
        return []
    else:
        segment_list = [MotorwaySegment(start, end, name, name_origin, source, crs) for start, end in
                        zip(point_list[:-1], point_list[1:])]
        return segment_list


# In[ ]:


def transform_segment(segment, transformer, to_crs):
    start = segment.start
    end = segment.end
    name = segment.name
    source = segment.source
    crs = segment.crs
    start_transformed = transformer.transform(start.y, start.x)
    end_transformed = transformer.transform(end.y, end.x)
    transformed_segment = MotorwaySegment(Point(*start_transformed), Point(*end_transformed), name, source, to_crs)
    return transformed_segment


# In[ ]:


def write_to_pickle(variable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)

def read_from_pickle(filename):
    with open(filename, 'rb') as f:
        variable = pickle.load(f)
    return variable


# In[ ]:


# Define Filepath and OSM tags
tags = {"highway":"motorway"}
motorway_file_path = Path(r"./data/motorways_osm.geojson")

# Download OSM Data to GeoDataFrame() if GeoJSON does not exist
if motorway_file_path.is_file():
    motorways = gpd.read_file(motorway_file_path)
else:
    # Needs special requirements for shapely, geopandas and other packages /!\
    # Use ox package to download geometries from
    import osmnx as ox
    motorways = ox.geometries_from_place("Switzerland",tags=tags)
    # Convert lists to strings for GeoJSON
    motorways = motorways.apply(lambda col: col.astype(str) if isinstance(col[0], list) else col)
    motorways.to_file(motorway_file_path, "GeoJSON")


# In[ ]:


#Remove unwanted segments and remove NA
discard_list = ["A1a", "A1H", "A1R","A1R;5", "A1L", "A3W", "A24", "A50", "7;A50", "A51"]
filter_osm = ~motorways.ref.isin(discard_list)
motorways = motorways[filter_osm]
motorways = motorways[motorways['ref'].notna()]


# In[ ]:


mway = motorways.copy()
mway["geometry"] = mway.geometry.explode(index_parts=False)
mway["source"] = "osm"
mway["crs"] = 'epsg:4326'
mway["name"] = mway['ref'].apply(name_mapping)

mway["segments"] = mway.apply(linestring_to_motorwaysegment_list, axis=1)
mway = mway.explode("segments", index_parts=False)
mway = mway.reset_index()

# Example 4326 to 2056
from_crs = 'epsg:4326'
to_crs = 'epsg:2056'

transformer = Transformer.from_crs(from_crs, to_crs)
motorway_segments_transformed = mway["segments"].apply(transform_segment, transformer=transformer, to_crs=to_crs)
    
# Create Lookup dictionary with Segments by segment name (1:[MotorwaysSegment,...],2:[],...)
segments_by_name = {}
for segment in motorway_segments_transformed:
    if segment.name not in segments_by_name:
        segments_by_name[segment.name] = []
    segments_by_name[segment.name].append(segment)

streetnames_osm = motorway_segments_transformed.apply(lambda x: x.name).unique()
motorway_osm_lines = []
for streetname in streetnames_osm:
    # Use streetname to retrieve list of segments
    list_segments = segments_by_name.get(streetname, [])
    if list_segments:
        motorwayLine = MotorwayLine(list_segments)
        motorway_osm_lines.append(motorwayLine)

motorway_osm_lines = sorted(motorway_osm_lines, key=lambda x: x.name)

#import .gdb
fiona.listlayers(Path(r"data/national_roads/ch.astra.nationalstrassenachsen.gdb"))
motorways_ch = gpd.read_file(Path(r"data/national_roads/ch.astra.nationalstrassenachsen.gdb"), driver='FileGDB', layer='Stammachsen')
if plot_steps:
    motorways_ch.plot()
    
# Remove positionscode
filt_equal = motorways_ch["Positionscode"] != "="
motorways_ch = motorways_ch.copy().loc[filt_equal]

# Extract streetname and convert to int
motorways_ch.loc[:, 'streetname'] = motorways_ch['Strassennummer'].str.extract(r'(\d+)').astype(int)

streetnames_ch = motorways_ch['streetname'].unique()
    
mway_ch = motorways_ch.copy()
mway_ch["geometry"] = mway_ch.geometry.explode(index_parts=False)
mway_ch["source"] = "astra"
mway_ch["crs"] = 'epsg:2056'
mway_ch["name"] = mway_ch['streetname']
mway_ch["ref"] = mway_ch["Strassennummer"]
mway_ch['segments'] = mway_ch.apply(linestring_to_motorwaysegment_list, axis=1)

segments_by_name = {}
for line in mway_ch['segments']:
    for segment in line:
        if segment.name not in segments_by_name:
            segments_by_name[segment.name] = []
        segments_by_name[segment.name].append(segment)

motorway_ch_lines = []
for streetname in streetnames_ch:
    list_segments = segments_by_name.get(streetname,[])
    if list_segments:
        motorwayLine = MotorwayLine(list_segments)
        motorway_ch_lines.append(motorwayLine)

motorway_ch_lines = sorted(motorway_ch_lines, key=lambda x: x.name)

# motorway_ch_lines to Pickle
write_to_pickle(motorway_osm_lines, "motorway_osm_lines.pkl")
write_to_pickle(motorway_ch_lines, "motorway_ch_lines.pkl")

