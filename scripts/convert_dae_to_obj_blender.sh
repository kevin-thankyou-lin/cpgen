#!/bin/bash

# Check if Blender is installed and working
if ! blender --version > /dev/null 2>&1; then
  echo "Blender is not installed or not in the system's PATH."
  exit 1
fi

echo "Blender is installed and working."

# Check if the folder path is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <folder_path>"
  exit 1
fi

# Folder containing .dae or .stl files
FOLDER_PATH=$1

# Check if the provided path is a directory
if [ ! -d "$FOLDER_PATH" ]; then
  echo "The specified path $FOLDER_PATH is not a directory."
  exit 1
fi


# Iterate over all .dae and .stl files in the folder
for FILE in "$FOLDER_PATH"/*.{dae,stl}; do
  if [ -f "$FILE" ]; then
    # Construct the output .obj file name
    OBJ_FILE="${FILE%.*}.obj"

    # Use Blender to convert the file to .obj
    blender --background --python-expr "
import bpy
import sys
# Function to remove default objects and objects without geometry
def clean_scene():
    for obj in bpy.data.objects:
        if obj.type in ['CAMERA', 'LIGHT'] or (obj.type == 'MESH' and len(obj.data.vertices) == 0):
            bpy.data.objects.remove(obj, do_unlink=True)
def set_correct_orientation():
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            # Select and make the object active
            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)

            # Apply any existing transformations
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Rotate 90 degrees around X-axis to get correct orientation
            obj.rotation_euler.x = 1.5707

            # Apply the rotation
            bpy.ops.object.transform_apply(rotation=True)
# Load the file
if '$FILE'.endswith('.dae'):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.collada_import(filepath='$FILE')
elif '$FILE'.endswith('.stl'):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_mesh.stl(filepath='$FILE')
else:
    sys.exit('Unsupported file format')
# Clean the scene
clean_scene()
set_correct_orientation()
# Export to .obj
bpy.ops.export_scene.obj(filepath='$OBJ_FILE')
" && echo "Converted $FILE to $OBJ_FILE" || echo "Error converting $FILE"
  fi
done
