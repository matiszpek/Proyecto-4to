import ezdxf

def extract_features(file_path):
    # Load DXF file
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    
    points, lines, features = [], [], []
    
    for entity in msp:
        if entity.dxftype() == 'POINT':
            points.append((entity.dxf.location.x, entity.dxf.location.y, entity.dxf.location.z))
        elif entity.dxftype() == 'LINE':
            lines.append(((entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z),
                          (entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z)))
        else:
            features.append(entity)

    return points, lines, features

# Example usage:
file_path = "dwg_tran\Part 1 Drawing 1.dxf"
points, lines, features = extract_features(file_path)

# Print results
print("Points:", points)
print("Lines:", lines)
print("Other Features:", features)
