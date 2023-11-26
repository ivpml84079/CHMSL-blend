from os import listdir
import csv

csvfile = open("files.csv")
path = list(csv.reader(csvfile, delimiter=','))
files = listdir(path[0][0])
print(files)
csvfile.close()

with open('files.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    csvfile.truncate()
    writer.writerow(files)
csvfile.close()
