from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from detect import init
from detect import detect
from parl.utils import logger
import os
import time

# Create your views here.
params = init()

people_num=0
battery_remaining_level=0
bandwidth_quality=0
local_people_confidence_sum=0
local_process_time=0

def objectdetect(request):
    print("接受请求")
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
        boxes = results['boxes']
        confidence_sum = results['confidence_sum']
        people_num = results['people_num']
        # server_process_time = results['server_process_time']
        os.remove(os.path.join(origin_image_path, fileImage.name))
        json_dict = {"code": 200, "boxes": boxes, "ServerPeopleConfidenceSum": confidence_sum, "PeopleNum": people_num}
        return JsonResponse(json_dict)
        # return JsonResponse(boxes, safe=False)
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
    people_num=int(request.POST.get("PeopleNum"))
    battery_remaining_level = float(request.POST.get("BatteryRemainingLevel"))
    bandwidth_quality = float(request.POST.get("BandwidthQuality"))
    local_people_confidence_sum = float(request.POST.get("LocalPeopleConfidenceSum"))
    local_process_time = float(request.POST.get("LocalProcessTime"))
    logger.info('people_num:{}    battery_remaining_level:{}    bandwidth_quality:{}    local_people_confidence_sum:{}    local_process_time:{}'.format(
        people_num, battery_remaining_level, bandwidth_quality, local_people_confidence_sum, local_process_time))
    return HttpResponse('ok')

def testBandwidthState(request):
    return HttpResponse('网络带宽状态获取完成')

def getPeopleConfidenceSum():
    return people_confidence_sum

def getLocalProcessTime():
    return local_process_time

