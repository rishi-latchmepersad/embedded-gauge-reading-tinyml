path = 'ml/src/embedded_gauge_reading_tinyml/geometry_crop_dataset.py'
with open(path, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "loose_crop_x1=int(float(row['loose_crop_x1'])," in line:
        lines[i] = line.replace(
            "loose_crop_x1=int(float(row['loose_crop_x1']),",
            "loose_crop_x1=int(float(row['loose_crop_x1'])),"
        )
    if "loose_crop_y1=int(float(row['loose_crop_y1'])," in line:
        lines[i] = line.replace(
            "loose_crop_y1=int(float(row['loose_crop_y1']),",
            "loose_crop_y1=int(float(row['loose_crop_y1'])),"
        )
    if "loose_crop_x2=int(float(row['loose_crop_x2'])," in line:
        lines[i] = line.replace(
            "loose_crop_x2=int(float(row['loose_crop_x2']),",
            "loose_crop_x2=int(float(row['loose_crop_x2'])),"
        )
    if "loose_crop_y2=int(float(row['loose_crop_y2'])," in line:
        lines[i] = line.replace(
            "loose_crop_y2=int(float(row['loose_crop_y2']),",
            "loose_crop_y2=int(float(row['loose_crop_y2'])),"
        )

with open(path, 'w') as f:
    f.writelines(lines)

print('Fixed parentheses')
