import json
import torch
from utils.datasets import letterbox
from utils.torch_utils import select_device, load_classifier, time_synchronized
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from numpy import random
import cv2, numpy as np
import os
from cnocr import CnOcr

ocr = CnOcr(det_model_name='en_PP-OCRv3_det', rec_model_name='en_PP-OCRv3')

device = 'cpu'
classes = None
agnostic_nms = False
conf_thres = 0.1
iou_thres = 0.45
augment = False
imgsz = 640
weights1 = 'weights/receipt_v2.pt'
weights2 = 'weights/transactions.pt'
device = select_device(device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model1 = attempt_load(weights1, map_location=device)  # load FP32 model
model2 = attempt_load(weights2, map_location=device)  # load FP32 model

names1 = model1.module.names if hasattr(model1, 'module') else model1.names
names2 = model2.module.names if hasattr(model2, 'module') else model2.names


def take_first(tup):
    return tup[0][0]


def sort_boxes(txt_boxes, all_texts):
    all_boxes_unsorted = []
    all_texts_unsorted = []
    fun_boxes = []
    fun_texts = []
    for tb in range(len(list(txt_boxes))):

        box_txt = list(txt_boxes)[tb]
        text_ori = all_texts[tb]
        if len(fun_boxes) == 0 and len(fun_texts) == 0:
            fun_boxes.append(box_txt)
            fun_texts.append(text_ori)
        try:
            next_box_txt = list(txt_boxes)[tb + 1]
            next_text_ori = all_texts[tb + 1]

            y_box = box_txt[1][1]
            nxy_box = next_box_txt[1][1]

            if abs(y_box - nxy_box) < 55:
                fun_boxes.append(next_box_txt)
                fun_texts.append(next_text_ori)
            else:
                all_boxes_unsorted.append(fun_boxes)
                all_texts_unsorted.append(fun_texts)
                fun_boxes = [next_box_txt]
                fun_texts = [next_text_ori]
        except:
            all_boxes_unsorted.append(fun_boxes)
            all_texts_unsorted.append(fun_texts)
    # print(all_texts_unsorted)
    # text_triggers = []
    final_boxes = []
    for all in all_boxes_unsorted:
        # trigga_j = []
        # for a in all:
        #     trigga_j.append(a[0][0])
        # text_triggers.append(trigga_j)
        all = sorted(all, key=take_first)
        final_boxes.append(all)

    final_texts = []
    for f in range(len((final_boxes))):
        all_texts_sorted = []
        for all in range(len(final_boxes[f])):
            box_look = final_boxes[f][all]
            means = [np.mean(absrt) for absrt in all_boxes_unsorted[f]]
            # print(means)
            # print(np.mean(box_look))

            ind = means.index(np.mean(box_look))
            tex = all_texts_unsorted[f][ind]
            # print(tex)
            all_texts_sorted.append(tex)
        # print(all_texts_sorted)
        final_texts.append(all_texts_sorted)
    # print(final_texts)
    # for tx in range(len(text_triggers)):
    #     text_trigga = text_triggers[tx]
    #     text_list = all_texts_unsorted[tx]
    #     text_list = [x for _, x in sorted(zip(text_trigga, text_list))]
    #     final_texts.append(text_list)
    #
    sorted_boxes = []
    sorted_texts = []
    for i in range(len(final_boxes)):
        sorted_boxes += final_boxes[i]
        sorted_texts += final_texts[i]

    return sorted_boxes, sorted_texts, final_texts, final_boxes


def get_instances(box, full_image, fresh_image, label, tr_extractor, out_dir):
    tr_full = fresh_image.copy()
    least_height = 0
    most_height = 0
    qt_x1 = 0
    qt_x2 = 0
    it_x1 = 0
    it_x2 = 0
    pr_x1 = 0
    pr_x2 = 0
    if label == 'dt' or label == 'id':
        image_ext = fresh_image[box[1]:box[3], box[0]: box[2]]
    elif label == 'tr':
        image_ext = tr_full[box[1]:box[3], 0: box[2]]
        img_to_draw = image_ext.copy()
        box = [0, box[1], box[2], box[3]]
        tr_boxes = detect_tr(image_ext, tr_extractor[0], 640, tr_extractor[1])

        heights = []
        for key, val in tr_boxes.items():
            if len(val) > 0:
                heights.append(val[1])
                heights.append(val[3])
                least_height = min(heights)
                most_height = max(heights)
                if key == 'qt':
                    qt_x1 = val[0]
                    qt_x2 = val[2]
                if key == 'it':
                    it_x1 = val[0]
                    it_x2 = val[2]
                if key == 'pr':
                    pr_x1 = val[0]
                    pr_x2 = val[2]
                # imm = image_ext[val[1]: val[3], val[0]: val[2]]
                # cv2.imwrite(f'transactions/{key}_{int_tr}.jpg', imm)
                # cv2.rectangle(img_to_draw, (val[0], val[1]), (val[2], val[3]), (0, 0, 255), 1)
        #         int_tr += 1

    else:
        image_ext = fresh_image[box[1]:box[3],
                    box[0] - 100 if box[0] - 100 > 0 else 0:
                    box[2] + 100 if box[2] + 100 < fresh_image.shape[0] else fresh_image.shape[0]]
        box = [box[0] - 100 if box[0] - 100 > 0 else 0, box[1],
               box[2] + 100 if box[2] + 100 < fresh_image.shape[0] else fresh_image.shape[0], box[3]]

    img = f'{out_dir}/{label}.jpg'
    cv2.imwrite(img, image_ext)
    out_raw = ocr.ocr(img)

    # if label == 'tr':
    #     for o in out_raw:
    #         array = o['position']
    #         img_to_draw = cv2.polylines(img_to_draw, [array.astype(int)],
    #                               True, [255, 0, 0], 1)
    #     cv2.imwrite(f'static/transactions/{label}.jpg', img_to_draw)

    out = []

    if label == 'tr':
        for otraw in out_raw:
            if (otraw['position'][0][1] + otraw['position'][3][1])/2 > least_height or \
               (otraw['position'][0][1] + otraw['position'][3][1])/2 < most_height:
                out.append(otraw)
    else:
        out = out_raw


    texts_unsorted = [o['text'] for o in out]
    out_trigger = [((int(o['position'][0][1] * 1.6666 + o['position'][1][1] * 1.6666)) / 2 * 1.764111 +
                    (int(o['position'][2][1] * 1.6666 + o['position'][3][1] * 1.6666)) / 2 * 1.764111) / 2
                   for o in out]
    scores = [o['score'] for o in out]
    texts = [x for _, x in sorted(zip(out_trigger, texts_unsorted))]
    scores_sorted = [x for _, x in sorted(zip(out_trigger, scores))]
    texts = list(filter(None, texts))

    scores_sorted = [i for i in scores_sorted if not np.isnan(i)]
    boxes_all = []
    for t in scores_sorted:
        ind = scores.index(t)
        box_one = out[ind]['position']
        boxes_all.append(box_one)
    boxes_all = np.array(boxes_all)

    boxes_all, texts, texts_list, boxes_list = sort_boxes(boxes_all, texts)
    # if label == 'tr':
    #     print(boxes_list, texts_list)
    txt = f''
    tl = round(0.002 * (fresh_image.shape[0] + fresh_image.shape[1]) / 2) + 1
    tf = max(tl - 1, 1)

    for li in range(len(boxes_all)):
        lines = boxes_all[li]
        txt_single = texts[li]
        frame = full_image.copy()
        lines_update = np.array([[int(line[0] + box[0]), int(line[1] + box[1])] for line in lines])
        t_size = cv2.getTextSize(txt_single, 0, fontScale=tl / 3, thickness=tf)[0]
        c1 = (int(lines_update[0][0]), int(lines_update[0][1]))
        c2 = int(lines_update[0][0]) + t_size[0], int(lines_update[0][1]) - t_size[1] - 3
        cv2.rectangle(frame, c1, c2, [0, 0, 255], -1, cv2.LINE_AA)
        txt += f'{txt_single} '
        cv2.fillPoly(frame, [lines_update],
                     [0, 0, 255])
        full_image = cv2.addWeighted(frame, 0.2, full_image, 1 - 0.2, gamma=0)
        cv2.putText(full_image, txt_single, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)

    tr_data = (least_height, most_height, qt_x1, qt_x2, it_x1, it_x2, pr_x1, pr_x2)
    return full_image, txt, texts_list, boxes_list, tr_data


def detect(img0, model, imgsz, names):
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    img = letterbox(img0, 640, stride=stride)[0]
    
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    pred = model(img, augment=augment)[0]
    
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    
    id_flag = False
    dt_flag = False
    tr_flag = False
    br_flag = False
    
    boxes = {'id': (), 'dt': (), 'tr': (), 'br': ()}
    
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            for *xyxy, conf, cls in det:
                label = f'{names[int(cls)]} {conf:.2f}'
                # print(label)
                lbl = label.split(' ')[0]
                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())
                if lbl == 'id' and id_flag == False:
                    boxes['id'] = (x1, y1, x2, y2)
                    id_flag = True
                elif lbl == 'dt' and dt_flag == False:
                    boxes['dt'] = (x1, y1, x2, y2)
                    dt_flag = True
                elif lbl == 'tr' and tr_flag == False:
                    boxes['tr'] = (x1, y1, x2, y2)
                    tr_flag = True
                elif lbl == 'br' and br_flag == False:
                    boxes['br'] = (x1, y1, x2, y2)
                    br_flag = True
            
    return boxes


def detect_tr(img0, model, imgsz, names):
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    img = letterbox(img0, 640, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=augment)[0]

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

    qt_flag = False
    it_flag = False
    pr_flag = False

    boxes_tr = {'qt': (), 'it': (), 'pr': ()}
    # Process detections
    for i, det in enumerate(pred):  # detections per image

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in det:
                label = f'{names[int(cls)]} {conf:.2f}'
                # print(label)
                lbl = label.split(' ')[0]
                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())
                if lbl == 'qt' and qt_flag == False:
                    boxes_tr['qt'] = (x1, y1, x2, y2)
                    qt_flag = True
                elif lbl == 'it' and it_flag == False:
                    boxes_tr['it'] = (x1, y1, x2, y2)
                    it_flag = True
                elif lbl == 'pr' and pr_flag == False:
                    boxes_tr['pr'] = (x1, y1, x2, y2)
                    pr_flag = True

    return boxes_tr


def main(img_path):
    flage = False
    try:
        colors1 = [[random.randint(0, 255) for _ in range(3)] for _ in names1]
        
        # for root,dirs,files in os.walk('agumintation'):
        #
        #     for file in files:
        img_path = img_path
        image_name = img_path.split('.')[0]
        image_name = image_name.split('/')
        image_name = image_name[2]
        print(image_name)
        file = f'{image_name}.jpg'
        out_dir = f'static/result/{image_name}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        img0 = cv2.imread(img_path)
        fresh_img = img0.copy()
        boxes = detect(img0, model1, imgsz, names1)
    
        final_out = {}
        id_box = boxes['id']
        if id_box != ():
            img0, id_txt, _, _, _ = get_instances(id_box, img0, fresh_img, 'id', None, out_dir)
            final_out['Shop ID'] = id_txt
            sh_id = id_txt
        else:
            print('id box not detected')
            sh_id = ''
            
        dt_box = boxes['dt']
        if dt_box != ():
            img0, dt_txt, _, _, _ = get_instances(dt_box, img0, fresh_img, 'dt', None, out_dir)
            final_out['Date & Time'] = dt_txt
            d_t = dt_txt
            flage = True
        else:
            print('dt box not detected')
            d_t = ''
    
        tr_box = boxes['tr']
        if tr_box != ():
            img_tr = fresh_img[tr_box[1]:tr_box[3], 0:tr_box[2]]
            cv2.imwrite(f'static/result/{file}', img_tr)
            img0, _, tr_txt, tr_boxes, tr_data = get_instances(tr_box, img0, fresh_img, 'tr', (model2, names2), out_dir)
            (least_height, most_height, qt_x1, qt_x2, it_x1, it_x2, pr_x1, pr_x2) = tr_data
            items = {}
            for ii in range(len(tr_boxes)):
                boxy = tr_boxes[ii]
                txty = tr_txt[ii]
                quantity = ''
                item = ''
                price = ''
                for jj in range(len(boxy)):
                    bbx = boxy[jj]
                    txt = txty[jj]
                    point_x = (bbx[0][0] + bbx[1][0]) / 2
                    point_y = (bbx[0][1] + bbx[3][1]) / 2
        
                    if point_y < least_height or point_y > most_height:
                        continue
                    else:
                        if point_x < qt_x2 and qt_x1 != 0 and qt_x2 != 0:
                            quantity += f'{txt}'
                        elif point_x > it_x1 and point_x < it_x2 and it_x1 != 0 and it_x2 != 0:
                            item += f'{txt}'
                        elif point_x > pr_x1 and pr_x1 != 0 and pr_x2 != 0:
                            price += f'{txt}'
                if item == '':
                    continue
                else:
                    full_item = {'Quantity': quantity if quantity != '' else '1', 'Item': item, 'Price': price}
                    items[f'Entry_{ii+1}'] = full_item
            transaction = str(items)
            final_out['Transaction'] = items
        else:
            print('tr box not detected')
            transaction = ''
        
        br_box = boxes['br']
        if br_box != ():
            img0, br_txt, _, _, _ = get_instances(br_box, img0, fresh_img, 'br', None, out_dir)
            final_out['Barcode'] = br_txt
            barcode = br_txt
            flage = True
        else:
            print('br box not detected')
            barcode = ''
            
        with open(f'{out_dir}/results.json', 'w') as f:
            json.dump(final_out, f, indent=4)
        dict_path = f'{out_dir}/results.json'
        out_ticket = np.concatenate((fresh_img, img0), axis=1)
        cv2.imwrite(f'static/result/{file}', out_ticket)
        
        return flage, dict_path,sh_id,d_t,transaction,barcode
    except:
        return flage, {}

# path = 'static/input_img/IMG_0399.jpg'
# main(path)