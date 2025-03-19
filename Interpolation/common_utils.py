import os
import shutil

def move_images(src, dest):
    for filename in os.listdir(src):
        if filename.endswith('.png'):
            source_file = os.path.join(src, filename)
            target_file = os.path.join(dest, filename)
            shutil.copy(source_file, target_file)