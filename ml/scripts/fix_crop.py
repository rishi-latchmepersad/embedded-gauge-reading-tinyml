path = "ml/src/embedded_gauge_reading_tinyml/geometry_crop_dataset.py"
with open(path, "r") as f:
    content = f.read()

content = content.replace("loose_crop_x1=int(row[", "loose_crop_x1=int(float(row[")
content = content.replace("loose_crop_y1=int(row[", "loose_crop_y1=int(float(row[")
content = content.replace("loose_crop_x2=int(row[", "loose_crop_x2=int(float(row[")
content = content.replace("loose_crop_y2=int(row[", "loose_crop_y2=int(float(row[")

with open(path, "w") as f:
    f.write(content)

print("Fixed")
