import os

category_id = ['厨余垃圾','可回收物','其他垃圾','有害垃圾']
paths = os.listdir('./data/new_img')
i = 0
for path in paths:
    try:
        label = str(category_id.index(path.split('_')[0]))+'_'+str(i)
        os.rename('./data/new_img/'+path,'./data/new_img/'+label)
    except:
        label = path
    files = os.listdir('./data/new_img/'+label)
    i += 1
    j = 1
    for f in files:
        name = 'img'+str(j)
        j += 1
        try:
            os.rename('./data/new_img/'+label+'/'+f,'./data/new_img/'+label+'/'+name)
        except:
            pass
        print('./data/new_img/'+label+'/'+name)
