import pandas as pd
from pathlib import Path

manifests = [
    'data/canonical_manifest_v1.csv',
    'data/unified_training_manifest_v1.csv', 
    'data/full_labelled_plus_board30_valid_with_new5.csv',
    'data/hard_cases_plus_board30_valid_with_new6.csv',
    'data/new_labelled_captures4.csv',
    'data/all_captured_images_manifest.csv',
    'data/full_scalar_manifest_v1.csv',
]

all_paths = set()
for m in manifests:
    p = Path(m)
    if p.exists():
        df = pd.read_csv(p)
        col = 'image_path' if 'image_path' in df.columns else 'path'
        for ip in df[col]:
            all_paths.add(Path(ip).name)

print(f'Total unique image filenames across all manifests: {len(all_paths)}')

captured_dir = Path('data/captured_images')
if captured_dir.exists():
    pngs = list(captured_dir.glob('*.png'))
    jpgs = list(captured_dir.glob('*.jpg'))
    print(f'PNG files in captured_images: {len(pngs)}')
    print(f'JPG files in captured_images: {len(jpgs)}')
    print(f'Total image files: {len(pngs) + len(jpgs)}')
