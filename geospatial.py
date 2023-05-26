# Classes and methods for geospatial algorithms based on points with x, y coordinates

# for GEO877 Spatial Algorithms
# - requires Python 3.6 or later (or replace f-string with older print format)
# - Point distance measures return results in metres (or assume metres)

# release history:
# 2021-03-27, v1.0: updated version of points.py (v1.0) for solution 1 (comparing distance measures)
# author:  Sharon Richardson
# version: 1.0

# 2022-03-25, v2.1: Modified Sharon's original polygon code slightly
# author:  Ross Purves


# 2022-04-02, v2.2: Added Segment class and determinants to Point class
# author:  Ross Purves
# version: 2.2
# --------------------------

from numpy import sqrt, radians, arcsin, sin, cos

# class and methods for a geometric point
# =======================================
from numpy import sqrt


####### Point #######

class Point():
    # initialise
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
    
    # representation
    def __repr__(self):
        return f'Point(x={self.x}, y={self.y})'

        # Test for equality between Points
    def __eq__(self, other): 
        if not isinstance(other, Point):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.x == other.x and self.y == other.y

    # We need this method so that the class will behave sensibly in sets and dictionaries
    def __hash__(self):
        return hash((self.x, self.y))
    
    # calculate Euclidean distance between two points
    def distEuclidean(self, other):
        return sqrt((self.x-other.x)**2 + (self.y-other.y)**2)
    
    # calculate Manhattan distance between two points
    def distManhattan(self, other):
        return abs(self.x-other.x) + abs(self.y-other.y)

    # Haversine distance between two points on a sphere - requires lat/lng converted to radians
    def distHaversine(self, other):
        r = 6371000  # Earth's radius in metres (will return result in metres)
        phi1 = radians(self.y) # latitudes
        phi2 = radians(other.y)
        lam1 = radians(self.x) # longitudes
        lam2 = radians(other.x)

        d = 2 * r * arcsin(sqrt(sin((phi2 - phi1)/2)**2 + 
                                      cos(phi1) * cos(phi2) * sin((lam2 - lam1)/2)**2))
        return d   

    
    # Calculate determinant with respect to three points. Note the order matters here - we use it to work out left/ right in the next method
    def __det(self, p1, p2):
        det = (self.x-p1.x)*(p2.y-p1.y)-(p2.x-p1.x)*(self.y-p1.y)       
        return det

    def leftRight(self, p1, p2):
        # based on GIS Algorithms, Ch2, p11-12, by Ningchuan Xiao, publ. 2016
        # -ve: this point is on the left side of a line connecting p1 and p2
        #   0: this point is collinear
        # +ve: this point is on the right side of the line
        side = int(self.__det(p1, p2))
        if side != 0:
            side = side/abs(side)  # will return 0 if collinear, -1 for left, 1 for right
        return side

####### Segment #######
class Segment():    
    # initialise
    def __init__(self, p0, p1):
        self.start = p0
        self.end = p1
        self.length = p0.distEuclidean(p1)
    
    # representation
    def __repr__(self):
        return f'Segment with start {self.start} and end {self.end}.' 

    # Test for equality between Segments - we treat segments going in opposite directions as equal here
    def __eq__(self, other): 
        if (self.start == other.start or self.start == other.end) and (self.end == other.end or self.end == other.start):
            return True
        else:
            return False
            # We need this method so that the class will behave sensibly in sets and dictionaries
    
    def __hash__(self):
        return hash((self.start.x, self.start.y,self.end.x, self.end.y))    
    
    
    
    # determine if intersects with another segment (using Point method leftRight)
    # - should we incorporate testing for identical segments and non-zero lengths?
    def intersects(self, other):        
        
        # first test if projections overlap 
        
        mnx = min(other.start.x, other.end.x)
        mny = min(other.start.y, other.end.y)
        mxx = max(other.start.x, other.end.x)
        mxy = max(other.start.y, other.end.y)
        
        if max(self.start.x,self.end.x) < mnx:
            return False
        if max(self.start.y,self.end.y) < mny: 
            return False
        if min(self.start.x,self.end.x) > mxx:
            return False
        if min(self.start.y,self.end.y) > mxy: 
            return False
        
        # If we get here, projections do overlap

        # find side for each point of each line in relation to the points of the other line
        # Positive det means both left or both right -> no intersection
        abp = self.start.leftRight(self.end, other.start)
        abq = self.start.leftRight(self.end, other.end)
        if abp * abq > 0:
            return False

        # We only calculate these determinants if we need to
        pqa = other.start.leftRight(other.end, self.start)
        pqb = other.start.leftRight(other.end, self.end)           
        if pqa * pqb > 0:
            return False
        
        return True
    
####### Bounding Box #######    

# define bounding box for 2 or more points (initialised as PointGroup, Polygon or Segment)
class Bbox():
    
    # initialise
    def __init__(self, data):
    # using built-in `isinstance` to test what class has been used to initialise the object   

        # for Segment objects 
        if isinstance(data, Segment) == True:
            x = [data.start.x, data.end.x]
            y = [data.start.y, data.end.y]
        
        # for PointGroup objects (including Polygon)
        else:      
            x = [i.x for i in data]   # extract all x coords as a list
            y = [i.y for i in data]   # extract all y coords as a list

        # determine corners, calculate centre and area
        self.ll = Point(min(x), min(y))    # lower-left corner (min x, min y)
        self.ur = Point(max(x), max(y))    # upper-right corner (max x, max y)
        self.ctr = Point((max(x)-min(x))/2, (max(y)-min(y))/2)   # centre of box
        self.area = (abs(max(x)-min(x)))*abs((max(y)-min(y)))    # area of box
           
    # representation
    def __repr__(self):
        return f'Bounding box with lower-left {self.ll} and upper-right {self.ur}' 
    
    def __eq__(self, other): 
        if (self.ll == other.ll and self.ur == other.ur):
            return True
        else:
            return False
            # We need this method so that the class will behave sensibly in sets and dictionaries
    
    def __hash__(self):
        return hash(self.ll, other.ur)  
        
    # test for overlap between two bounding boxes
    def intersects(self, other):       
        # test if bboxes overlap (touching is not enough to be compatible with the approach to segments)
        if (self.ur.x > other.ll.x and other.ur.x > self.ll.x and
            self.ur.y > other.ll.y and other.ur.y > self.ll.y):
            return True
        else:
            return False
    def containsPoint(self, p):
        if (self.ur.x > p.x and p.x > self.ll.x and
            self.ur.y > p.y and p.y > self.ll.y):
            return True
        return False
        

# FOR GROUPS OF POINTS

# data provided should be as an array of points with x, y coordinates.

# class for a group of Points, assumes initial data is unsorted, spatially
class PointGroup(): 
    # initialise
    def __init__(self, data=None, xcol=None, ycol=None):
        self.points = []
        self.size = len(data)
        for d in data:
            self.points.append(Point(d[xcol], d[ycol]))
    
    # representation
    def __repr__(self):
        return f'PointGroup containing {self.size} points' 
 
    # create index of points in group for referencing
    def __getitem__(self, key):
        return self.points[key]
    

# Polygon class for polygons, assumes initial data is in a spatially sorted order
class Polygon(PointGroup):  
    # initialise
    def __init__(self, data=None, xcol=None, ycol=None):
        self.points = []
        self.size = len(data)
        for d in data:
            self.points.append(Point(d[xcol], d[ycol]))
        self.bbox = Bbox(self)
        
    # representation
    def __repr__(self):
        return f'Polygon PointGroup containing {self.size} points' 
  
    # test if polygon is closed: first and last point should be identical
    def isClosed(self):
        start = self.points[0]
        end = self.points[-1]
        return start == end

    def removeDuplicates(self):
        oldn = len(self.points)
        self.points = list(dict.fromkeys(self.points)) # Get rid of the duplicates
        self.points.append(self.points[0]) # Our polygon must have one duplicate - we put it back now
        n = len(self.points)
        print(f'The old polygon had {oldn} points, now we only have {n}.')
        
        # find area and centre of the polygon
    # - based on GIS Algorithms, Ch.2 p9-10, by Ningchuan Xiao, publ. 2016 
    
    def __signedArea(self):  # used for both area and centre calculations - this is a private method (only used within the class)
        a = 0
        xmean = 0
        ymean = 0
        for i in range(0, self.size-1):
            ai = self[i].x * self[i+1].y - self[i+1].x * self[i].y
            a += ai
            xmean += (self[i+1].x + self[i].x) * ai
            ymean += (self[i+1].y + self[i].y) * ai

        a = a/2.0   # signed area of polygon (can be a negative)
    
        return a, xmean, ymean
    
    def area(self):
        a, xmean, ymean = self.__signedArea()
        area = abs(a)   # absolute area of polygon
        
        return area

    def centre(self):
        a, xmean, ymean = self.__signedArea() # note we use the signed area here
        centre = Point(xmean/(6*a), ymean/(6*a)) # centre of polygon 
        return centre

    def containsPoint(self, p):
        if (self.bbox.containsPoint(p) == False):
            return False
        
        # Solution, as discussed in lecture, added here
        ray = Segment(p, Point(self.bbox.ur.x+1, p.y))
        count = 0
        
        for i in range(0, self.size-1):
            start = self[i]
            end = self[i+1]
            s = Segment(start, end)
            if s.intersects(ray):
                if (p.y != min(start.y, end.y)):
                    count = count + 1


                
        #print(f'count: {count}')
        #print(p)
        if (count%2 == 0):
            return False           

        return True