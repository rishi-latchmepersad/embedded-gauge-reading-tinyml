path = 'ml/scripts/train_geometry_heatmap_v1.py'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    if '# Save history' in lines[i]:
        new_lines.append('    # Save history\n')
        new_lines.append('    print("\\nSaving history...")\n')
        new_lines.append('    history_data = history.history\n')
        new_lines.append('    with open(output_dir / "history.csv", "w") as f:\n')
        new_lines.append('        f.write("epoch,loss,center_heatmap_loss,tip_heatmap_loss,confidence_loss,val_loss,val_center_heatmap_mae,val_tip_heatmap_mae,lr\\n")\n')
        new_lines.append('        num_epochs = len(history_data.get("loss", []))\n')
        new_lines.append('        for j in range(num_epochs):\n')
        new_lines.append('            loss = history_data.get("loss", [0]*num_epochs)[j]\n')
        new_lines.append('            chm_loss = history_data.get("center_heatmap_loss", [0]*num_epochs)[j]\n')
        new_lines.append('            thm_loss = history_data.get("tip_heatmap_loss", [0]*num_epochs)[j]\n')
        new_lines.append('            conf_loss = history_data.get("confidence_loss", [0]*num_epochs)[j]\n')
        new_lines.append('            val_loss = history_data.get("val_loss", [0]*num_epochs)[j]\n')
        new_lines.append('            val_chm_mae = history_data.get("val_center_heatmap_mae", [0]*num_epochs)[j]\n')
        new_lines.append('            val_thm_mae = history_data.get("val_tip_heatmap_mae", [0]*num_epochs)[j]\n')
        new_lines.append('            lr = history_data.get("learning_rate", [0]*num_epochs)[j]\n')
        new_lines.append('            f.write(f"{epoch},{loss},{chm_loss},{thm_loss},{conf_loss},{val_loss},{val_chm_mae},{val_thm_mae},{lr}\\n")\n')
        while i < len(lines) and 'print(f"Saved history to' not in lines[i]:
            i += 1
        i += 1
    else:
        new_lines.append(lines[i])
        i += 1

with open(path, 'w') as f:
    f.writelines(new_lines)

print('Fixed history saving')
