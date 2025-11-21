import json
import os

# Define file names
input_file = 'audiocaps_val.json'
output_file = 'audiocaps_val_fixed.json'
bad_file = 'AudioCaps_val.json'

# 1. Fix the good file (audiocaps_val.json)
print(f"Reading {input_file}...")
try:
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    fixed_count = 0
    for item in data:
        # Fix the broken path
        if item.get('file_path') == "WHAT_IS_THIS?":
            sound_name = item.get('sound_name')
            if sound_name:
                item['file_path'] = f"audio/val/{sound_name}"
                fixed_count += 1
    
    print(f"Fixed {fixed_count} paths.")
    
    # Write the fixed data to a new file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved fixed data to {output_file}")
    
    # 2. Rename/Backup the bad file (AudioCaps_val.json) to prevent it from being loaded
    if os.path.exists(bad_file):
        backup_name = bad_file + ".bak"
        os.rename(bad_file, backup_name)
        print(f"Renamed bad file {bad_file} to {backup_name}")
    else:
        print(f"Bad file {bad_file} not found (already removed?)")

    # 3. Rename fixed file to original name
    os.rename(output_file, input_file)
    print(f"Replaced {input_file} with fixed version.")
    
    print("\nSUCCESS! You can now run your training/eval again.")

except FileNotFoundError:
    print(f"Error: Could not find {input_file}. Make sure you are in the 'audiocaps' directory.")
except Exception as e:
    print(f"An error occurred: {e}")
