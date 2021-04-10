from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from detect import init
from detect import detect
import os

# Create your views here.
params = init()

def objectdetect(request):
    # params=init();
    # detect(params)
    fileImage = request.FILES.get("fileImage")
    # print(fileImage)

    if fileImage:
        origin_image_path = os.path.join("/software/ObjectDetectionProject/data/images")
        w = open(os.path.join(origin_image_path, fileImage.name), 'wb+')
        for chunk in fileImage.chunks():
            w.write(chunk)
        w.close()
        '''if type == "False":
            params = init()
        else:
            detect(params)'''
        results = detect(params)
        os.remove(os.path.join(origin_image_path, fileImage.name))
        return JsonResponse(results, safe=False)
        '''print("imageName=" + fileImage.name)
        result_image_path = os.path.join("/software/ObjectDetectionProject/runs/detect/exp/" + fileImage.name)
        # result_label_path = os.path.join("/software/ObjectDetectionProject/runs/detect/exp/labels/" + fileImage.name[0:-4] + 'txt')
        print("result_image_path=" + str(result_image_path))
        image_data = open(result_image_path, "rb").read()
        os.remove(os.path.join(origin_image_path, fileImage.name))
        # os.remove(result_image_path)
        return HttpResponse(image_data, content_type="image/jpg")'''
        # return HttpResponse('ok')
    else:
        return HttpResponse('failure')

def trainDQN(request):
    pass