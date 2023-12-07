import torchvision.utils as vutils
import torchvision.transforms as standard_transforms
import cv2
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os


def save_results_more(iter, exp_path, img0, pre_map0, gt_map0, pre_cnt, gt_cnt, pre_points=None,gt_points=None, pre_seg_map0=None, gt_seg_map0=None):  # , flow):
    # gt_cnt = gt_map0.sum().item()
    # pre_cnt = pre_map0.sum().item()
    pil_to_tensor = standard_transforms.ToTensor()
    tensor_to_pil = standard_transforms.ToPILImage()
    
    UNIT_H , UNIT_W = img0.size(1), img0.size(2)
    
    img0 = img0.detach().to('cpu')
    pre_map0 = pre_map0.detach().to('cpu')
    gt_map0 = gt_map0.detach().to('cpu')
    
    pre_map0 =  F.interpolate(pre_map0.unsqueeze(0),size=(UNIT_H,UNIT_W)).squeeze(0).numpy()
    gt_map0  =  F.interpolate(gt_map0.unsqueeze(0),size=(UNIT_H,UNIT_W)).squeeze(0).numpy()
    
    tensor = [img0, gt_map0, pre_map0]
    
    # 插值分割图
    if gt_seg_map0 is not None: 
        gt_seg_map0 = gt_seg_map0.float().detach().to('cpu')
        gt_seg_map0 =  F.interpolate(gt_seg_map0.unsqueeze(0),size=(UNIT_H,UNIT_W)).squeeze(0).numpy()
    if pre_seg_map0 is not None: 
        pre_seg_map0 = pre_seg_map0.float().detach().to('cpu')
        pre_seg_map0 =  F.interpolate(pre_seg_map0.unsqueeze(0),size=(UNIT_H,UNIT_W)).squeeze(0).numpy()
    
    
    pil_input0 = tensor_to_pil(tensor[0])

    gt_color_map = cv2.applyColorMap((255 * tensor[1] / (tensor[1].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
    pre_color_map = cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
    
    gt_seg_map = None if gt_seg_map0 is None else gt_seg_map0.astype(np.uint8).squeeze(0) * 255
    pre_seg_map = None if pre_seg_map0 is None else pre_seg_map0.astype(np.uint8).squeeze(0) * 255 


    # mask_color_map = cv2.applyColorMap((255 * tensor[8]).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
    RGB_R = (255, 0, 0)
    RGB_G = (0, 255, 0)

    BGR_R = (0,0,255)  # BGR
    BGR_G = (0,255,0)  # BGR
    thickness = 3
    lineType = 4
    pil_input0 = np.array(pil_input0)
    # if boxes is not None:
    #     for i, box in enumerate(boxes, 0):
    #         wh_LeftTop = (box[0], box[1])
    #         wh_RightBottom = (box[2], box[3])
    #         # cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
    #         cv2.rectangle(pil_input0, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
    if pre_points is not None:
        for i, point in enumerate(pre_points, 0):
            # import pdb
            # pdb.set_trace()
            point = point.astype(np.int32)
            point = (point[0], point[1])
            cv2.drawMarker(pil_input0, point, RGB_G, markerType=cv2.MARKER_CROSS,markerSize=15,thickness=3)
            cv2.circle(pre_color_map, point,2, BGR_R,thickness)
            # cv2.drawMarker(pil_input0, point, RGB_R, markerType=cv2.MARKER,markerSize=20,thickness=3)

    if gt_points is not None:
        for i, point in enumerate(gt_points, 0):
            point = point.astype(np.int32)
            point = (point[0], point[1])
            cv2.circle(pil_input0, point,4, RGB_R,thickness)


    cv2.putText(gt_color_map, 'GT:'+str(gt_cnt), (100,150), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255,255,255),thickness=2)
    cv2.putText(pre_color_map, 'Pre:'+str(np.round(pre_cnt,1)), (100, 150),cv2.FONT_HERSHEY_SIMPLEX,
                3, (255,255,255), thickness=2)
    pil_input0 = Image.fromarray(pil_input0)

    pil_label0 = Image.fromarray(cv2.cvtColor(gt_color_map, cv2.COLOR_BGR2RGB))
    pil_output0 = Image.fromarray(cv2.cvtColor(pre_color_map, cv2.COLOR_BGR2RGB))

    imgs = [pil_input0, pil_label0, pil_output0]

    # 显示分割图
    if gt_seg_map is not None:
        # cv2.putText(gt_seg_map, 'GT', (100,150), cv2.FONT_HERSHEY_SIMPLEX, 3, RGB_R, thickness=2)
        pil_gt_seg0 = Image.fromarray(gt_seg_map)
        imgs.append(pil_gt_seg0)
    if pre_seg_map is not None:
        # cv2.putText(pre_seg_map, 'Pre', (100,150), cv2.FONT_HERSHEY_SIMPLEX, 3, RGB_R, thickness=2)
        pil_pre_seg0 = Image.fromarray(pre_seg_map)
        imgs.append(pil_pre_seg0)

    w_num , h_num=3, 2

    target_shape = (w_num * (UNIT_W + 10), h_num * (UNIT_H + 10))
    target = Image.new('RGB', target_shape)
    count = 0
    for img in imgs:
        x, y = int(count%w_num) * (UNIT_W + 10), int(count // w_num) * (UNIT_H + 10)  # 左上角坐标，从左到右递增
        target.paste(img, (x, y, x + UNIT_W, y + UNIT_H))
        count+=1

    # img = np.vstack([a,b])*255
    # import pdb
    # pdb.set_trace()
    # cv2.imwrite('./exp/{}_vis_.png'.format(iter), img)
    target.save(os.path.join(exp_path,'{}_den.jpg'.format(iter)))
    

def vis_results(exp_name, writer, img, pred_map, gt_map, binar_map, thresholds, boxes, steps):  # , flow):

    pil_to_tensor = standard_transforms.ToTensor()
    tensor_to_pil = standard_transforms.ToPILImage()
    x = []
    y = []

    for idx, tensor in enumerate(zip(img, pred_map, gt_map, binar_map, thresholds)):
        if idx > 1:  # show only one group
            break

        pil_input = tensor_to_pil(tensor[0])
        pred_color_map = cv2.applyColorMap((255 * tensor[1] / (tensor[1].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map = cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        binar_color_map = cv2.applyColorMap((255 * tensor[3] / (tensor[3].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        thresholds_color_map = cv2.applyColorMap((255 * tensor[4] / (tensor[4].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)



        point_color = (0, 255, 0)  # BGR
        thickness = 1
        lineType = 4
        pil_input = np.array(pil_input)
        # print(pil_input, binar_color_map)
        for i, box in enumerate(boxes, 0):
            wh_LeftTop = (box[0], box[1])
            wh_RightBottom = (box[0] + box[2], box[1] + box[3])
            # print(wh_LeftTop, wh_RightBottom)
            cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
            cv2.rectangle(pil_input, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)

        pil_input = Image.fromarray(pil_input)
        pil_label = Image.fromarray(cv2.cvtColor(gt_color_map, cv2.COLOR_BGR2RGB))
        pil_output = Image.fromarray(cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))
        pil_binar = Image.fromarray(cv2.cvtColor(binar_color_map, cv2.COLOR_BGR2RGB))
        pil_threshold = Image.fromarray(cv2.cvtColor(thresholds_color_map, cv2.COLOR_BGR2RGB))

        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_to_tensor(pil_label.convert('RGB')),
                  pil_to_tensor(pil_output.convert('RGB')), pil_to_tensor(pil_binar.convert('RGB')),
                  pil_to_tensor(pil_threshold.convert('RGB'))
                  ])
        # pdb.set_trace()  sum(sum(flow[0].cpu().data.numpy().transpose((1,2,0))[:,:,0]))
        # flow = flow[0].cpu().data.numpy().transpose((1,2,0))
        # flow0 = cv2.applyColorMap((255*flow[:,:,0]/(flow[:,:,0].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
        # flow1 = cv2.applyColorMap((255*flow[:,:,1]/(flow[:,:,1].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
        # flow2 = cv2.applyColorMap((255*flow[:,:,2]/(flow[:,:,2].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
        # flow0 = Image.fromarray(cv2.cvtColor(flow0,cv2.COLOR_BGR2RGB))
        # flow1 = Image.fromarray(cv2.cvtColor(flow1,cv2.COLOR_BGR2RGB))
        # flow2 = Image.fromarray(cv2.cvtColor(flow2,cv2.COLOR_BGR2RGB))
        # y.extend([pil_to_tensor(flow0.convert('RGB')), pil_to_tensor(flow1.convert('RGB')), pil_to_tensor(flow2.convert('RGB'))])

    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy() * 255).astype(np.uint8)

    # y = torch.stack(y,0)
    # y = vutils.make_grid(y,nrow=3,padding=5)
    # y = (y.numpy()*255).astype(np.uint8)

    # x = np.concatenate((x,y),axis=1)
    writer.add_image(exp_name, x, steps)