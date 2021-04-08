import json
import sqlite3
# import json2sql

jsonFile = "./data/annotations.json"
sqliteFile = "./data/data.sqlite"
logFile = "./data/json2sqlLog.txt"
    
jsonDict = json.load(open(jsonFile,"r"))
connection = sqlite3.connect(sqliteFile)
log = open(logFile,"w")

annotations = jsonDict["annotations"]
licenses = jsonDict["licenses"]
images = jsonDict["images"]
categories = jsonDict["categories"]

session = connection.cursor()
# session.execute("""
#     CREATE TABLE ANNONTATION(
#         ID INT PRIMARY  KEY     NOT NULL,
#         imageID         INT,
#         categoryID      INT,
#         segmentation    BLOB,
#         area            REAL,
#         bbox            BLOB
#         )
#     """)
# session.execute("""
#     CREATE TABLE IMAGES(
#         ID INT PRIMARY  KEY     NOT NULL,
#         width           INT,
#         height          INT,
#         fileName        TEXT,
#         license         TEXT,
#         flickrUrl       TEXT,
#         cocoUrl         TEXT,
#         dataCaptured    BLOB,
#         flickr640Url    TEXT
#         )
#     """)
# session.execute("""
#     CREATE TABLE CATEGORIES(
#         ID INT PRIMARY  KEY     NOT NULL,
#         supercategory   TEXT,
#         name            TEXT
#     )
# """)


# print("insert annotations.")
# for line in annotations:
#     ID = line["id"]
#     imagesID = line["image_id"]
#     categoryID = line["category_id"]
#     segmentation = line["segmentation"]
#     area = line["area"]
#     bbox = line["bbox"]
#     data = [ID,imagesID,categoryID,json.dumps(segmentation),area,json.dumps(bbox)]
#     try:
#         session.execute("INSERT INTO ANNONTATION VALUES (?,?,?,?,?,?)",data)
#     except Exception as e:
#         print("when INSERT INTO ANNONTATION :",data[0]," error\n",str(e),file=log)
#         continue
#     connection.commit()
#     print(data)
print("insert images.")
for line in images:
    data = [
        line["id"],
        line["width"],
        line["height"],
        line["file_name"],
        line["license"],
        line["flickr_url"],
        line["coco_url"],
        line["date_captured"],
        line["flickr_640_url"]
    ]
    try:
        session.execute("INSERT INTO IMAGES VALUES (?,?,?,?,?,?,?,?,?)", data)
    except Exception as e:
        print("when INSERT INTO IMAGES: ",data[0]," error\n",str(e),file=log)
        continue
    connection.commit()
    print(data)
print("insert categories.")
for line in categories:
    data = [
        line["id"],
        line["supercategory"],
        line["name"]
    ]
    try:
        session.execute("INSERT INTO CATEGORIES VALUES (?,?,?)", data)
    except Exception as e:
        print("when INSERT INTO CATEGORIES: ",data[0]," error\n",str(e),file=log)
        continue
    connection.commit()
    print(data)