import os
import xml.etree.ElementTree as ET

# Defines the class mapping (replace 'Oak_log' with your actual class name)
class_mapping = {'Oak_log': 0}

# Function to convert XML annotations to YOLO format
def convert_to_yolo(xml_content, target_width, target_height):
    root = ET.fromstring(xml_content)

    yolo_annotations = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = class_mapping.get(class_name)
        if class_id is None:
            raise ValueError(f"Class name '{class_name}' not found in the class mapping.")

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # Calculates bounding box center and width/height relative to image size
        x_center = (xmin + xmax) / (2 * target_width)
        y_center = (ymin + ymax) / (2 * target_height)
        width = (xmax - xmin) / target_width
        height = (ymax - ymin) / target_height

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return yolo_annotations

# Path to the folder containing XML annotations
xml_folder = "yolov7-main/Tree/Train/xmls"

# Target size for YOLO format
target_width, target_height = 1920, 1080

# Loops through all XML files in the folder
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(xml_folder, xml_file)

        # Read XML content
        with open(xml_path, 'r') as f:
            xml_content = f.read()

        print(f"Processing: {xml_file}")

        # Converts XML to YOLO format
        try:
            yolo_annotations = convert_to_yolo(xml_content, target_width, target_height)
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue

        # Writes YOLO annotations to a text file
        txt_file = xml_file.replace('.xml', '.txt')
        txt_path = os.path.join(xml_folder, txt_file)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

        print(f"Converted {xml_file} and saved YOLO annotations to {txt_file}")
