from google_images_download import google_images_download
import csv

keywords = []
with open('./data/categories.CSV') as f:
    labels = csv.reader(f)
    for row in labels:
        keywords.append(row[2])
del keywords[0]
total_num = 600

response = google_images_download.googleimagesdownload()
for i in keywords:
    arguments = {"keywords":i,"limit":20,"print_urls":True}
    absolute_image_paths = response.download(arguments)
    print(absolute_image_paths)