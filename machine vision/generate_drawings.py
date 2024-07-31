import random
from PIL import Image, ImageDraw
import numpy as np
PI = np.pi

class Line:
    def __init__(self, start, end, line_type, normal = None):
        self.start = start
        self.end = end
        self.midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        self.line_type = line_type  # 'drawing' or 'measurement'
        self.normal = normal

    def nomral_rad(self):
        if self.normal != None:
            return self.normal
        else:
            try:
                print(abs(self.end[1] - self.start[1])/abs(self.end[0] - self.start[0]))
                return abs(self.end[1] - self.start[1])/abs(self.end[0] - self.start[0])
            except ZeroDivisionError:
                print(self.end[1] - self.start[1], self.end[0] - self.start[0])
                return np.pi/2
    
    def slope_and_intercept(self):
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

def distance_between_points(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
            

def generate_random_point(max_x, max_y, min_x = 0, min_y = 0):
    return (random.randint(min_x, max_x), random.randint(min_y, max_y))

def generate_connected_boxes(num_boxes, max_x, max_y):
    lines = []
    start_point = generate_random_point(max_x, max_y)
    
    for _ in range(num_boxes):
        end_point = generate_random_point(max_x, max_y)
        n = [0, 0, 0, 0]
        if end_point[1] < start_point[1]:
            n[0] = 0
            n[2] = PI
        else:
            n[0] = PI
            n[2] = 0
        if end_point[0] < start_point[0]:
            n[1] = -PI/2
            n[3] = PI/2
        else:
            n[1] = PI/2
            n[3] = -PI/2

        lines.append(Line(start_point, (end_point[0], start_point[1]), 'drawing', n[0]))
        lines.append(Line((end_point[0], start_point[1]), end_point, 'drawing', n[1]))
        lines.append(Line(end_point, (start_point[0], end_point[1]), 'drawing', n[2]))
        lines.append(Line((start_point[0], end_point[1]), start_point, 'drawing', n[3]))
        start_point = end_point
    
    return lines

def generate_measurement_lines(num_lines, max_x, max_y, drawing_lines):
    lines = []
    for _ in range(num_lines):
        random_line = random.choice(drawing_lines)
        start_point = (random_line.start[0]+np.sin(random_line.nomral_rad())*25, random_line.start[1]+np.cos(random_line.nomral_rad())*25)
        end_point = (random_line.end[0]+np.sin(random_line.nomral_rad())*25, random_line.end[1]+np.cos(random_line.nomral_rad())*25)
        lines.append(Line(start_point, end_point, 'measurement', random_line.nomral_rad()))
        start_point = (random_line.start[0]+np.sin(random_line.nomral_rad())*10, random_line.start[1]+np.cos(random_line.nomral_rad())*10)
        end_point = (random_line.start[0]+np.sin(random_line.nomral_rad())*50, random_line.start[1]+np.cos(random_line.nomral_rad())*50)
        lines.append(Line(start_point, end_point, 'finish', random_line.nomral_rad()+PI/2))
        start_point = (random_line.end[0]+np.sin(random_line.nomral_rad())*10, random_line.end[1]+np.cos(random_line.nomral_rad())*10)
        end_point = (random_line.end[0]+np.sin(random_line.nomral_rad())*50, random_line.end[1]+np.cos(random_line.nomral_rad())*50)
        lines.append(Line(start_point, end_point, 'finish', random_line.nomral_rad()+PI/2))
    return lines

def draw_line(draw, line, color):
    draw.line([line.start, line.end], fill=color, width=2)

def draw_measurement(draw, line: Line, color):
    draw_line(draw, line, color)
    midpoint = ((line.start[0] + line.end[0]) // 2, (line.start[1] + line.end[1]) // 2)
    measurement = round(((line.end[0] - line.start[0])**2 + (line.end[1] - line.start[1])**2)**0.5, 2)
    if line.line_type == "measurement":
        draw.text((midpoint[0]+np.sin(line.nomral_rad())*25, midpoint[1]+np.cos(line.nomral_rad())*25), str(measurement), fill=color)

def generate_technical_drawing(width, height, num_drawing_lines, num_measurement_lines):
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    drawing_lines = generate_connected_boxes(num_drawing_lines, width, height)
    measurement_lines = generate_measurement_lines(num_measurement_lines, width, height, drawing_lines)
    
    for i, line in enumerate(drawing_lines):
        draw_line(draw, line, 'black')

    for i, line in enumerate(measurement_lines):
        draw_measurement(draw, line, 'red')
    
    return image, drawing_lines + measurement_lines

def save_image_and_data(image, lines, filename_prefix):
    image.save(f"{filename_prefix}.png")
    
    """with open(f"{filename_prefix}_data.txt", 'w') as f:
        for i, line in enumerate(lines):
            f.write(f"Line {i}: Start={line.start}, End={line.end}, Type={line.line_type}\n")"""

# Generate and save a sample image
for i in range(10):
    image, lines = generate_technical_drawing(800, 600, 2, 5)
    save_image_and_data(image, lines, "technical_drawing_sample"+str(i))

print("Sample image and data have been generated and saved.")