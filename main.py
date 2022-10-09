import os

# Run script for QS1 with RGB colorspace, all comparison methods
os.system('python week1.py -c rgb -s e -q False -f 1')
os.system('python week1.py -c rgb -s l -q False -f 2')
os.system('python week1.py -c rgb -s x -q False -f 3')

# Run script for QS1 with YCrCb colorspace, all comparison methods
os.system('python week1.py -c ycrcb -s e -q False -f 4')
os.system('python week1.py -c ycrcb -s l -q False -f 5')
os.system('python week1.py -c ycrcb -s x -q False -f 6')

# Run script for QS1 with HSV colorspace, all comparison methods
os.system('python week1.py -c hsv -s e -q False -f 7')
os.system('python week1.py -c hsv -s l -q False -f 8')
os.system('python week1.py -c hsv -s x -q False -f 9')

# Run script for QS1 with Average Grayscale, all comparison methods
os.system('python week1.py -c gray -g a -s e -q False -f 10')
os.system('python week1.py -c gray -g a -s l -q False -f 11')
os.system('python week1.py -c gray -g a -s x -q False -f 12')

# Run script for QS1 with Weighted Grayscale, all comparison methods
os.system('python week1.py -c gray -g w -s e -q False -f 13')
os.system('python week1.py -c gray -g w -s l -q False -f 14')
os.system('python week1.py -c gray -g w -s x -q False -f 15')


# Run script for QS2 with Weighted Grayscale
os.system('python week1.py -m gray -g w -q True -f 1')

# Run script for QS2 with RGB
os.system('python week1.py -m rgb -q True -f 2')

# Run script for QS2 with YCrCb
os.system('python week1.py -m ycrcb -q True -f 3')

# Run script for QS2 with HSV
os.system('python week1.py -m hsv -q True -f 4')

