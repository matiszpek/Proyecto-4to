import numpy as np
PI = np.pi
import cv2 as cv

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
    def __init__(self, start: tuple[int, int], end: tuple[int, int], line_type: str = None, normal: float = PI*1000):
        self.start = start
        self.end = end
        self.midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        self.line_type = line_type  # 'drawing' or 'measurement'
        if normal != PI*1000:
            self.normal = normal
        else:
            try:
                self.normal = abs(self.end[1] - self.start[1])/abs(self.end[0] - self.start[0])
            except ZeroDivisionError:
                self.normal = np.pi

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

def reduce_lines(lineReductions: int, lines_: list[Line]): # WIP
    """Reduces lines by removing similar lines"""
    for i in range(lineReductions):
            for line in lines_:
                p1, p2, mp, type, angle = line.get()
                for line_ in lines_:
                    if line == line_:   # Skip the same line
                        continue

                    p1_, p2_, mp_, type, angle_ = line_.get()
                    if PI/90 > abs(angle - angle_) >= 0:
                        if distance_between_lines_vectorized_normalized(line, line_)[0] < 10:
                            # print("Removed: ", line_)
                            # print("diff: ", abs(angle - angle_))
                            lines_.remove(line_)
                    else:
                        # print("Not Removed: ", line_)
                        # print("diff: ", abs(angle - angle_))
                        pass

def draw_line(img_lines: cv.typing.MatLike, line: Line, text: str = None, color: tuple[int, int, int] = (0, 255, 0)):
    """Draws line on image"""
    p1, p2, mp, type, ang = line.get()
    x1, y1 = p1
    x2, y2 = p2
    cv.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv.putText(img_lines, text, (int((x2+x1)/2), int((y2+y1)/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv.LINE_AA)
    # cv.circle(img_lines, (x1, y1), 5, (255, 0, 0), -1)
    # cv.circle(img_lines, (x2, y2), 5, (255, 255, 0), -1)

