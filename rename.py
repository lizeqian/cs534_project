# Run this file in the root directory of all the videos
# It will go throught all the sub directories starts from current directory
# Or you can modify path to an absolute path

import sys
import os

path = '.'
fileNameLength = 10

fileList = []

def getIndex(filename):
    filename = filename[:len(filename)-4]
    filename = filename[filename.find('_')+1:]
    filename = filename[filename.find('_')+1:]
    return filename

for (path, dirs, files) in os.walk(path):
    for filename in files:
        try:
            extension = filename[len(filename)-4:]
        except BaseException:
            continue
        if extension == '.png':
            newName = getIndex(filename)
            newName = newName.zfill(fileNameLength)
            newName += '.png'
            os.rename(path + "/" + filename, path + "/" + newName)
