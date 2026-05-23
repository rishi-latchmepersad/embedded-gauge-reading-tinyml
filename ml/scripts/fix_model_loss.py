path = 'ml/src/embedded_gauge_reading_tinyml/models_geometry.py'
with open(path, 'r') as f:
    content = f.read()

# Fix heatmap_loss to use MeanSquaredError class
old_heatmap = '''def heatmap_loss(y_true, y_pred):
        return keras.losses.mean_squared_error(y_true, y_pred)'''
        
new_heatmap = '''def heatmap_loss(y_true, y_pred):
        return keras.losses.MeanSquaredError()(y_true, y_pred)'''

# Fix confidence_loss to use BinaryCrossentropy class  
old_conf = '''def confidence_loss(y_true, y_pred):
        return keras.losses.binary_crossentropy(y_true, y_pred)'''
        
new_conf = '''def confidence_loss(y_true, y_pred):
        return keras.losses.BinaryCrossentropy()(y_true, y_pred)'''

content = content.replace(old_heatmap, new_heatmap)
content = content.replace(old_conf, new_conf)

with open(path, 'w') as f:
    f.write(content)

print('Fixed loss functions')
