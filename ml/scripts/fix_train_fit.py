path = 'ml/scripts/train_geometry_heatmap_v1.py'
with open(path, 'r') as f:
    content = f.read()

# Fix model.fit to use y_train directly as dict
old_fit = '''history = model.fit(
        X_train,
        {
            'center_heatmap': [th['center_heatmap'] for th in y_train],
            'tip_heatmap': [th['tip_heatmap'] for th in y_train],
            'confidence': [th['confidence'] for th in y_train],
        },
        validation_data=(
            X_val,
            {
                'center_heatmap': [th['center_heatmap'] for th in y_val],
                'tip_heatmap': [th['tip_heatmap'] for th in y_val],
                'confidence': [th['confidence'] for th in y_val],
            },
        ),'''
        
new_fit = '''history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),'''

content = content.replace(old_fit, new_fit)

with open(path, 'w') as f:
    f.write(content)

print('Fixed model.fit call')
