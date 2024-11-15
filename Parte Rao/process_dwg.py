import os
import ezdxf

def extract_line_data(dxf_path):
    # Try reading the DXF file
    try:
        doc = ezdxf.readfile(dxf_path)
    except IOError:
        print(f"File {dxf_path} not found.")
        return []
    except ezdxf.DXFStructureError:
        print(f"{dxf_path} is not a valid DXF file.")
        return []

    line_data = []
    # Iterate over entities in the modelspace
    for entity in doc.modelspace():
        if entity.dxftype() == "LINE":
            start = (entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z)
            end = (entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z)
            line_data.append({
                'type': 'LINE',
                'start': start,
                'end': end
            })
        elif entity.dxftype() == "CIRCLE":
            center = (entity.dxf.center.x, entity.dxf.center.y, entity.dxf.center.z)
            radius = entity.dxf.radius
            line_data.append({
                'type': 'CIRCLE',
                'center': center,
                'radius': radius
            })
        # Additional feature types can be added here.

    return line_data

def process_file(file_path):
    # Check if the file is a DWG or DXF
    if file_path.lower().endswith(".dwg"):
        print("DWG files are not supported yet. Support for DWG files will be added in the future.")
        return []

    if not file_path.lower().endswith(".dxf"):
        print(f"Unsupported file type: {file_path}. Only DXF files are supported.")
        return []

    # Extract data from the DXF file
    return extract_line_data(file_path)

# Example usage
file_path = "C:\\Users\\ADM\\Documents\\GitHub\\Proyecto-4to\\Parte Rao\\cube.dxf"  # Replace with the path to your DXF file or DWG file
features = process_file(file_path)
for feature in features:
    print(feature)
