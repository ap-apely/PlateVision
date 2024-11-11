import os

directory = './img'

max_char = 9
char = $

for filename in os.listdir(directory):
    if filename.endswith('.png'):
        name, ext = os.path.splitext(filename)
        
        if len(name) < max_char:
            name += char * (max_char - len(name))
        
        new_filename = name + ext
        
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

print("Rename ended!")
