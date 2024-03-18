import xmltodict
from skimage import io

def get_xml_path_from_image_path(image_path):
    image_name = image_path.split("/")[-1].split(".")[0]
    return "/home/daniel/repos/roads/datasets/train/annotations/xmls/"+image_name+".xml"

def get_yolo_path_from_image_path(image_path):
    yolo_path = image_path.replace("images", "annotations")
    yolo_path = yolo_path.rstrip(".jpg")
    yolo_path += ".txt"
    return yolo_path
    
def get_annotation_xml_from_image_path(image_path):
    annotation_path = get_xml_path_from_image_path(image_path)
    return get_annotation_xml(annotation_path)

def get_annotation_xml(annotation_path):
    with open(annotation_path, "r") as f:
        annotation = f.read()
    annotation = xmltodict.parse(annotation)
    return annotation["annotation"]

def get_simplified_annotation_for_image(image_path):
    annotation = get_annotation_xml_from_image_path(image_path)
    return get_simplified_annotation(annotation)

def get_simplified_annotation(annotation):
    size = annotation["size"]
    size = {k: int(size[k]) for k in size.keys() if size[k]}
    sa = {"size": size, "filename": annotation["filename"], "objects": []}    
    if not "object" in annotation.keys():
        return sa
    if type(annotation["object"]) != list:
        obj = [annotation["object"]]
    else:
        obj = annotation["object"]
    objects =  [{"class": o["name"], "coords": dict(o["bndbox"])} for o in obj]
    for ann in objects:
        for k in ann["coords"].keys():
            ann["coords"][k] = int(float(ann["coords"][k]))
    sa["objects"] =  objects
    return sa

def create_yolo_path_from_xml_path(xml_path):
    return xml_path.split(".")[0].replace("/xmls", "") + ".txt"

def create_yolo_annotation(xml_path):
    yolo_annotation_path = create_yolo_path_from_xml_path(xml_path)
    annotation = get_annotation_xml(xml_path)
    sa = get_simplified_annotation(annotation)
    image_width = sa["size"]["width"]
    image_height = sa["size"]["height"]
    classdict = {x: i for i, x in enumerate(["D00", "D10", "D20", "D30", "D40"])}
    to_write = ""
    for obj in sa["objects"]:
        c = classdict[obj["class"]]
        coords = obj["coords"]
        bbox_width = (coords["xmax"]-coords["xmin"])
        bbox_height = (coords["ymax"]-coords["ymin"])
        x = (coords["xmin"] + bbox_width) / image_width
        y = (coords["ymin"] + bbox_height) / image_height
        to_write+= f"{c} {x} {y} {bbox_width / image_width} {bbox_height / image_height}\n"
    with open(yolo_annotation_path, "w") as f:
        f.write(to_write)
    return yolo_annotation_path
    