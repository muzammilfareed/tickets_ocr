from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import glob
import os
from main import main
from corner_point import coner_point


@csrf_exempt
def index(request):
    folder = 'static/input_img/'
    if request.method == "POST" and request.FILES['file']:
        file = request.FILES.get('file')
        url = request.get_host()
        
        input_img = glob.glob('static/input_img/*')
        for f in input_img:
            os.remove(f)
        # result = glob.glob('static/result/*')
        # for f in result:
        #     os.remove(f)
        location = FileSystemStorage(location=folder)
        fn = location.save(file.name, file)
        path = os.path.join('static/input_img/', fn)
        cflage, image_path = coner_point(path)
        print(cflage)
        if cflage == True:
            flag, json_path,sh_id,d_t,transaction,barcode = main(image_path)
            if flag == True:
                # output_path = 'static/result/out_put.jpg'
                url_path = f'{url}/{json_path}'
                context = {
                    "flag":flag,
                    "status": "Image text read successfully!",
                    "json_path": url_path,
                    "sh_id":sh_id,
                    "d_t":d_t,
                    "transaction":transaction,
                    "barcode":barcode,
                }
            else:
                context = {
                    "flag": flag,
                    "status": "please send the valid image",
                }
            return JsonResponse(context)
        else:
            context = {
                "flag": cflage,
                "status": "coner point not deduct",
            }
        return JsonResponse(context)
    return render(request, 'index.html')
    