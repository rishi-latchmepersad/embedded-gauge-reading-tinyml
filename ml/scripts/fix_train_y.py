path = 'ml/scripts/train_geometry_heatmap_v1.py'
with open(path, 'r') as f:
    content = f.read()

# Fix the return to convert y list to dict of arrays
old_return = 'return np.array(X), y'
new_return = '''# Convert y list to dict of arrays
    y_dict = {
        'center_heatmap': np.array([item['center_heatmap'] for item in y]),
        'tip_heatmap': np.array([item['tip_heatmap'] for item in y]),
        'confidence': np.array([item['confidence'] for item in y]),
    }
    return np.array(X), y_dict'''

content = content.replace(old_return, new_return)

with open(path, 'w') as f:
    f.write(content)

print('Fixed y return format')
