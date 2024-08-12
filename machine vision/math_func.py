import numpy as np
PI = np.pi
import cv2 as cv
from typing import Union, Optional, Tuple
import math

# Point functions
def distance_between_points(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """returns distance between two points"""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def move_point(p: tuple[int, int], angle: float, distance: float) -> tuple[int, int]:
    """moves point p by distance in angle"""
    return (p[0] + distance * np.cos(angle), p[1] + distance * np.sin(angle))

def rotate_point(p: tuple[int, int], angle: float, center: tuple[int, int]) -> tuple[int, int]:
    """rotates point p around center by angle"""
    x, y = p
    x_c, y_c = center
    x -= x_c
    y -= y_c
    x_, y_ = x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)
    return (x_ + x_c, y_ + y_c)

def vectorized_distance_between_points(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[float, float]:
    return (p1[0] - p2[0], p1[1] - p2[1])

def get_midpoint(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int]:
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def get_normal(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    try:
        return abs(p2[1] - p1[1])/abs(p2[0] - p1[0])
    except ZeroDivisionError:
        return np.pi

# Line functions
class Line:
    """Line class"""
    def __init__(self, 
                 start_or_line: Union['Line', Tuple[int, int]], 
                 end: Optional[Tuple[int, int]] = None, 
                 line_type: Optional[str] = None, 
                 normal: Optional[float] = None):
        """create a line by pasing starting values or another line"""
        if isinstance(start_or_line, Line):
            # If a Line object is passed, copy its attributes
            self.start = start_or_line.start
            self.end = start_or_line.end
            self.midpoint = start_or_line.midpoint
            self.n_dist = start_or_line.n_dist
            self.line_type = start_or_line.line_type
            self.normal = start_or_line.normal
        else:
            # If tuples for start and end points are passed
            self.start = start_or_line
            self.end = end
            self.midpoint = ((self.start[0] + self.end[0]) // 2, (self.start[1] + self.end[1]) // 2)
            self.n_dist = (self.end[0] - self.start[0], self.end[1] - self.start[1])
            self.line_type = line_type  # 'drawing' or 'measurement'
            
            if normal is not None:
                self.normal = normal
            else:
                if abs(self.end[0] - self.start[0]) != 0:
                    self.normal = abs(self.end[1] - self.start[1]) / abs(self.end[0] - self.start[0])
                else:
                    self.normal = PI  

    def nomral_rad(self) -> float:
        """returns normal of line in radians"""
        if self.normal != None:
            return self.normal
        else:
            try:
                print(abs(self.end[1] - self.start[1])/abs(self.end[0] - self.start[0]))
                return abs(self.end[1] - self.start[1])/abs(self.end[0] - self.start[0])
            except ZeroDivisionError:
                print(self.end[1] - self.start[1], self.end[0] - self.start[0])
                return np.pi/2
    
    def slope_and_intercept(self) -> tuple[float, float]:
        """returns slope and intercept of line"""
        x1, y1 = self.start
        x2, y2 = self.end
        x = np.sort([x1, x2])
        y = np.sort([y1, y2])
        x1, x2 = x
        y1, y2 = y
        x2 = x2 if x1 != x2 else x2+0.5
        slope = (y2 - y1) / (x2 - x1)
        slope = slope if slope != 0 else 0.05
        b = y1 - slope * x1
        return slope, b
    
    def get(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], str, float]:
        """returns start, end, midpoint, line_type, normal"""
        return self.start, self.end, self.midpoint, self.line_type, self.normal

    def lerp(self, n: float) -> tuple[int, int]:
        """lerp function"""
        x = self.start[0] + self.n_dist[0]*n
        y = self.start[1] + self.n_dist[1]*n
        return int(x), int(y)

    def to_general_form(self) -> Tuple[float, float, float]:
        """Convert the line to its general form: Ax + By = C."""
        x1, y1 = self.start
        x2, y2 = self.end
        A = y2 - y1
        B = x1 - x2
        C = A * x1 + B * y1
        return A, B, C

    def swap_ends(self):
        """Swap the start and end points of the line."""
        self.start, self.end = self.end, self.start
        self.midpoint = ((self.start[0] + self.end[0]) // 2, (self.start[1] + self.end[1]) // 2)
        self.n_dist = (self.end[0] - self.start[0], self.end[1] - self.start[1])
        self.normal = abs(self.end[1] - self.start[1]) / abs(self.end[0] - self.start[0]) if self.end[0] != self.start[0] else PI

def distance_between_lines(line1: Line, line2: Line) -> float:
    """returns distance between two lines"""
    return distance_between_points(line1.midpoint, line2.midpoint)

def distance_between_lines_vectorized(line1: Line, line2: Line) -> tuple[float, float]:
    """returns vectorized distance between two lines"""
    return vectorized_distance_between_points(line1.midpoint, line2.midpoint)

def distance_between_lines_vectorized_normalized(line1: Line, line2: Line) -> tuple[float, float]:
    """returns normalized distance between two lines"""
    rp1 = rotate_point(line2.midpoint, -line1.normal, line1.midpoint)
    return vectorized_distance_between_points(line1.midpoint, rp1)

def correct_angles(lines_: list[Line]) -> list[Line]:
    for line in lines_:
        angle = line.normal
        if angle > PI/2:
            angle = -PI + angle
            line.normal = angle
        elif angle < -PI/2:
            angle = PI + angle
            line.normal = angle
    return lines_

def reduce_lines(lineReductions: int, lines_: list[Line], img_: cv.typing.MatLike, line_threshold: int, check_points: int = 2) -> tuple[list[Line], list[Line]]: # WIP
    """
    Reduces by removing similar lines
    
    returns passed lines and deleted lines
    """
    
    # img = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
    deleted_lines = []
    for i in range(lineReductions):
            for line in lines_:
                p1, p2, mp, type, angle = line.get()
                for line_ in lines_:
                    if line == line_:   # Skip the same line
                        continue

                    p1_, p2_, mp_, type, angle_ = line_.get()
                    if PI/90 > abs(angle - angle_) >= 0:
                        if distance_between_lines_vectorized_normalized(line, line_)[0] < 10:
                            ml = Line(mp, mp_)
                            n = 0
                            for i in range(check_points):
                                p = ml.lerp(i/check_points)
                                if img_[p[1], p[0]] <= line_threshold:
                                    n += 1
                            if n >= check_points/2:
                                lines_.remove(line_)
                                print("removed line:", line_)
                                deleted_lines.append(Line(line_))
                    else:
                        # print("Not Removed: ", line_)
                        # print("diff: ", abs(angle - angle_))
                        pass
    return lines_, deleted_lines

def draw_line(img_lines: cv.typing.MatLike, line: Line, text: str = None, color: tuple[int, int, int] = (0, 255, 0)):
    """Draws line on image"""
    p1, p2, mp, type, ang = line.get()
    x1, y1 = p1
    x2, y2 = p2
    cv.line(img_lines, (x1, y1), (x2, y2), color, 1)
    cv.putText(img_lines, text, (int((x2+x1)/2), int((y2+y1)/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv.LINE_AA)
    # cv.circle(img_lines, (x1, y1), 5, (255, 0, 0), -1)
    # cv.circle(img_lines, (x2, y2), 5, (255, 255, 0), -1)

def check_point_in_box(p: tuple[int, int], box: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    """checks if point is in box"""
    x, y = p
    x1, y1 = box[0]
    x2, y2 = box[1]
    return x1 <= x <= x2 and y1 <= y <= y2

def infinite_line_intersection(line1: Line, line2: Line) -> Optional[Tuple[float, float]]:
    """Check if two infinite lines intersect, and find the intersection point if they do."""
    A1, B1, C1 = line1.to_general_form()
    A2, B2, C2 = line2.to_general_form()
    
    # Calculate the determinant
    det = A1 * B2 - A2 * B1
    
    if det == 0:
        # Lines are parallel (no intersection)
        return None
    
    # Intersection point
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    
    return x, y

def segment_infinite_intersection(segment: Line, infinite_line: Line) -> Optional[Tuple[float, float]]:
    """Check if a segment and an infinite line intersect, and find the intersection point if they do."""
    A1, B1, C1 = segment.to_general_form()
    A2, B2, C2 = infinite_line.to_general_form()
    
    # Calculate the determinant
    det = A1 * B2 - A2 * B1
    
    if det == 0:
        # Lines are parallel (no intersection)
        return None
    
    # Intersection point
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    
    # Check if the intersection point is within the segment's bounds
    if min(segment.start[0], segment.end[0]) <= x <= max(segment.start[0], segment.end[0]) and \
       min(segment.start[1], segment.end[1]) <= y <= max(segment.start[1], segment.end[1]):
        return x, y
    else:
        return None

def line_intersect(line1: Line, line2: Line, extend: Tuple[bool, bool] = (False, False)) -> Optional[tuple[float, float]]:
    """returns intersection point between line1 adn line 2, or None if they don't intersect"""
    if extend == (False, False):
        xdiff = (line1.start[0] - line1.end[0], line2.start[0] - line2.end[0])
        ydiff = (line1.start[1] - line1.end[1], line2.start[1] - line2.end[1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(line1.start, line1.end), det(line2.start, line2.end))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return (x, y)
    
    elif extend[0] == False:
        return segment_infinite_intersection(line1, line2)

    elif extend[1] == False:
        return segment_infinite_intersection(line2, line1)
    
    else:
        return infinite_line_intersection(line1, line2)
    
def check_line_passes_box(line: Line, box: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    """checks if line passes through box"""
    m, b = line.slope_and_intercept()

    x1, y1 = line.start
    x2, y2 = line.end
    box_left, box_top = box[0]
    box_right, box_bottom = box[1]

    # Check if the line passes through diagonalls of the box
    if line_intersect(line, Line((box_left, box_top), (box_right, box_bottom))) | line_intersect(line, Line((box_right, box_top), (box_left, box_bottom))):
        return True
    # Check if the line is inside the box
    elif box_left <= x1 <= box_right and box_left <= x2 <= box_right and box_top <= y1 <= box_bottom and box_top <= y2 <= box_bottom:
        return True
    else:
        return False

def join_borders(line1: Line, line2: Line) -> Tuple[Line, Line]:
    ip = line_intersect(line1, line2, (True, True))
    points = (line1.start, line1.end, line2.start, line2.end)
    dst = []
    for i in range(2):
        for n in range(2):
            dst.append(distance_between_points(points[i-1], points[1+n]))
    m = dst.index(dst.min())
    if m == 0:
        return Line(points[0], ip), Line(points[1], ip)
    elif m == 1:
        return Line(points[0], ip), Line(points[2], ip)
    elif m == 2:
        return Line(points[3], ip), Line(points[2], ip)
    elif m == 3:
        return Line(points[3], ip), Line(points[1], ip)
    
def is_point_close_line(line: Line, point: Tuple[int, int], threshold: float) -> bool:
    """Check if a point is close to the line within a given threshold."""
    x0, y0 = point
    A, B, C = line.to_general_form()
    
    distance = abs(A * x0 + B * y0 - C) / math.sqrt(A**2 + B**2)
    return distance <= threshold

# i need an adaptative version of that
def check_closest_points(p1: tuple[float, float], p_list: list[tuple[float, float]], points_returned: int) -> tuple[list[float], list[tuple[float, float]], list[int]]:
    """returns the distances to the closest points, their positions and their original IDs in the passed list"""

    dst = []
    p_ranked = p_list
    ID = np.arange(p_list.__len__)

    for p in p_list:
        dst.append(distance_between_points(p1, p))
    
    sorted_indices = np.argsort(dst)
    dst = dst[sorted_indices]
    p_ranked = p_ranked[sorted_indices]
    ID = ID[sorted_indices]

    return (dst[:points_returned], p_ranked[:points_returned], ID[:points_returned])

def lines_to_point_connections(lines: list[Line]) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    points = []
    for i, line in enumerate(lines):
        points.append(line.start)
        points.append(line.end)
    
    return (points)

def shift_array(_list: list[any], times: int = 1) -> list[any]:
    
    op_list = _list
    if times > 0:
        for i in range(times):
            n_list = []
            n_list.append(op_list[len(op_list)-1])
            for _ in op_list[:(len(op_list)-1)]:
                n_list.append(_)
            op_list = n_list
            
    elif times < 0:
        for i in range(-times):
            n_list = []
            n_list = op_list[1:]
            n_list.append(op_list[0])
            op_list = n_list
    else:
        return _list
    return n_list

def manage_opens(lines: list[Line], img: cv.typing.MatLike) -> tuple[list[Line], list[tuple[tuple[int, bool], tuple[int, bool]]]]:
    """connects all unconected lines and returns a new set with all of them conected and with the IDs of all the conected lines"""
    points, connections = lines_to_point_connections(lines)

    raise NotImplementedError

def get_lines(can):
    return cv.HoughLinesP(
        can,
        0.5,
        np.pi/8,
        10,
        minLineLength=5,
        maxLineGap=10
    )
        
def get_avrg_ligth(contour: list, img: cv.typing.MatLike) -> float:
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv.drawContours(mask, [contour], 0, (255), -1)

    masked_image = cv.bitwise_and(img, img, mask=mask)

    non_zero_pixels = masked_image[np.nonzero(masked_image)]
    if len(non_zero_pixels) > 0:
        average = np.mean(non_zero_pixels)
    else:
        average = 0
    return average

def rotate_image(image: cv.typing.MatLike, angle: float) -> cv.typing.MatLike:
    """Rotate an image by a given angle."""
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

def transform_cordinate_frame(cord: tuple[float, float], frame1: tuple[int, int], frame2: tuple[int, int]) -> tuple[int, int]:
    t_m = (frame1[0] / frame2[0], frame1[1] / frame2[1])
    return int(cord[0]*t_m[0]), int(cord[1]*t_m[1])

def lines_to_points(lines: list[Line]) -> list[tuple[int, int]]:
    points = []
    for line in lines:
        points.append(line.start)
        points.append(line.end)
    return points

def get_points_presence(points: list[tuple[int, int]], og_img: cv.typing.MatLike | tuple[int, int], res: tuple[int, int]) -> list[bool]:
    """returns an map with resolution res with the presence of points"""

    if isinstance(og_img, cv.typing.MatLike):
        w, h = og_img.shape[:2]
    else:
        w, h = og_img
    
    presence = np.zeros((int(res[0]), int(res[1]), 1), dtype=np.float64)
    point_weigth = 256/points.__len__()
    for point in points:
        x, y = transform_cordinate_frame(point, res, (w, h))
        presence[y, x] += point_weigth
    
    presence = cv.normalize(presence, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return presence

def get_line_presence(lines: list[Line], og_img: cv.typing.MatLike | tuple[int, int], res: tuple[int, int]) -> list[bool]:
    """returns an map with resolution res with the presence of lines"""

    if isinstance(og_img, cv.typing.MatLike):
        w, h = og_img.shape[:2]
    else:
        w, h = og_img
    
    presence = np.zeros((int(res[0]), int(res[1]), 1), dtype=np.float64)
    line_weigth = 256/lines.__len__()
    for line in lines:
        x1, y1 = transform_cordinate_frame(line.start, res, (w, h))
        x2, y2 = transform_cordinate_frame(line.end, res, (w, h))
        cv.line(presence, (x1, y1), (x2, y2), (line_weigth), 2)
    
    presence = cv.normalize(presence, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return presence

def scan_line_pixels(line: Line, img: cv.typing.MatLike) -> list[int]:
    """returns a list of pixels from a line in an image"""
    pixels = []
    x1, y1 = line.start
    x2, y2 = line.end
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1

    if dx > dy:
        err = dx / 2.0
        while x != x2:
            pixels.append(img[y, x])
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            pixels.append(img[y, x])
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return pixels