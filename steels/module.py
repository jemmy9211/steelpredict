import torch
import io, os, cv2, threading, argparse, logging, math, codecs, json
import pandas as pd
import numpy as np
import detectron2, shutil
from datetime import datetime
from viewer import PicViewer
from collections import UserDict
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from scipy.spatial import distance, distance_matrix
from imantics import Mask
from PIL import Image
from Report import writepdf
import webbrowser

def instance2mask(instances, colors):
    h, w = instances.image_size
    mask = np.zeros((h, w, 3), np.uint8)
    for c, m in zip(instances.pred_classes, instances.pred_masks):
        if colors[c]:
            m = cv2.cvtColor(m.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            mask = np.where(m == [True, True, True], colors[c], mask)
    return mask


def draw_masks(bg, masks):
    mask = bg.copy()
    for m in masks:
        mask = np.where(m == [0, 0, 0], mask, m)

    return mask.astype(np.uint8)


class SpacingModule(object):
    def __init__(self, threshold=0.8, device='cuda', model_path="./mrcnn_model/model_spacing_v1.pth", requests='./dataset/requests.xlsx'):
        cfg = get_cfg()
        cfg.draw_mask = True
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.threshold = threshold
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.DEVICE = device
        self.labels = ['Intersection', 'Spacing']
        self.colors = [(0, 0, 200), (0, 200, 0)]
        self.model = DefaultPredictor(cfg)
        self.requests = requests
        self.output_dir = './output'
        self.tolerance_ratio = 0.1
        self.min_num_for_active_inspection = 3
        self.tolerance_bias = 2
        self.max_pixel_dist = 15
        self.max_real_dist = 2

    def predict_mask(self, color, depth):
        depth_gray = np.stack(
            (cv2.cvtColor(depth, cv2.COLOR_BGR2HSV)[:, :, 0],)*3, axis=-1)
        weighted = cv2.addWeighted(color, 1, depth_gray, 1, 0)
        predictions = self.model(weighted)
        instances = predictions["instances"].to("cpu")
        mask = instance2mask(instances, self.colors)
        return mask, instances

    def predict_spacing(self, instances, coord, name):
        classes = instances.pred_classes.cpu()
        masks = np.asarray(instances.pred_masks.cpu())
        points = [[] for i in range(2)]
        centroids = []
        for c, mask in zip(classes, masks):
            polygons = Mask(mask).polygons()
            if not polygons.points: continue
            pts = []
            for contour in polygons.points:
                pts.extend(contour)
            points[c].append(pts)
            if c == 0:
                M = cv2.moments(np.array(pts))
                centroids.append((int(M['m10'] / M["m00"]), int(M['m01'] / M["m00"])))

        info = {'name': name, 'spacing': []}
        if len(centroids) < 2:
            return info, {"basename": name, "parameter": {}, "statistics": {'num_spacing': 0, 'num_group': 0, 'group_max': [],
            'group_min': [], 'group_mean': [], 'group_std': []}, "intersection": {}, "spacing": {}}

        for spacing_pts in points[1]:
            dist_mat = distance_matrix(spacing_pts, spacing_pts)
            i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
            dst_i = [distance.cdist(np.array(spacing_pts)[
                                    [i, ]], intersection_pts, 'euclidean').min() for intersection_pts in points[0]]
            dst_j = [distance.cdist(np.array(spacing_pts)[
                                    [j, ]], intersection_pts, 'euclidean').min() for intersection_pts in points[0]]

            id_i = np.argsort(dst_i)[0]
            id_j = np.argsort(dst_j)[0]
            if dst_i[id_i] < self.max_pixel_dist and dst_j[id_j] < self.max_pixel_dist:
                pt1 = centroids[id_i]
                pt2 = centroids[id_j]
                coord1 = coord[pt1[1], pt1[0]]
                coord2 = coord[pt2[1], pt2[0]]

                if np.sum(coord1 != 0) and np.sum(coord2 != 0) and np.linalg.norm(coord1) < self.max_real_dist and np.linalg.norm(coord2) < self.max_real_dist:
                    length = np.linalg.norm(coord1-coord2) * 100
                    dpx, dpy = pt1[0]-pt2[0], pt1[1]-pt2[1]
                    if dpx: theta = abs(math.atan(dpy / dpx) * 180 / math.pi)
                    else: theta = 90
                    info['spacing'].append({
                        'id': [id_i, id_j],
                        'points': [pt1, pt2],
                        'length': length,
                        'orientation': theta})
        result = self.inspect_info(info)
        return info, result

    def inspect_info(self, info):
        name = info['name']
        info = info['spacing']
        # Grouping spacings
        pairs = [IF['id'] for IF in info]
        intersection = {}
        for IF in info:
            for id, point in zip(IF['id'], IF['points']):
                if 'I'+str(id).zfill(3) not in intersection:
                    intersection['I'+str(id).zfill(3)] = [int(point[0]), int(point[1])]
        intersection = sorted(intersection.items(), key=lambda x: x[0])
        result = {"basename": name, "parameter": {}, "statistics": {},
            "intersection": intersection, "spacing": {}}
        groups = {}
        for (x, y) in pairs:
            xset = groups.get(x, set([x]))
            yset = groups.get(y, set([y]))
            jset = xset | yset
            for z in jset:
                groups[z] = jset
        groups = set(map(tuple, groups.values()))
        for i in range(len(info)):
            for j, group in enumerate(groups):
                if info[i]['id'][0] in group or info[i]['id'][1] in group:
                    info[i]['group_id'] = j

        # Sorting orientation
        groups = {i: [] for i in range(len(groups))}
        for IF in info:
            groups[IF['group_id']].append({
                'id': ['I'+str(id).zfill(3) for id in IF['id']],
                'points': IF['points'],
                'length': IF['length'],
                'orientation': IF['orientation'],
                'active_pass': True,
                'passive_pass': True})
        for gi, group in groups.items():
            groups[gi] = sorted(group, key=lambda k: k['orientation'])

        # Active & Passive inspection
        rows = pd.read_excel(self.requests).values.tolist()
        inspect_item = '鋼筋尺寸間距'
        requests = np.array([float(row[row.index(inspect_item)+1].split('@')
                            [-1])/10 for row in rows if inspect_item in row])
        statistics = {'num_spacing': 0, 'num_group': 0, 'group_max': [],
            'group_min': [], 'group_mean': [], 'group_std': []}
        for gi, group in groups.items():
            split_idx = 0
            statistics['num_spacing'] += len(group)
            for i in range(len(group)-1):
                if abs(group[i]['orientation'] - group[i+1]['orientation']) > 45:
                    split_idx = i + 1
                    break
            if split_idx:
                statistics['num_group'] += 2
                statistics['group_max'].append(
                    np.max([info['length'] for info in group[:split_idx]]))
                statistics['group_min'].append(
                    np.min([info['length'] for info in group[:split_idx]]))
                statistics['group_mean'].append(
                    np.mean([info['length'] for info in group[:split_idx]]))
                statistics['group_std'].append(
                    np.std([info['length'] for info in group[:split_idx]]))
                statistics['group_max'].append(
                    np.max([info['length'] for info in group[split_idx:]]))
                statistics['group_min'].append(
                    np.min([info['length'] for info in group[split_idx:]]))
                statistics['group_mean'].append(
                    np.mean([info['length'] for info in group[split_idx:]]))
                statistics['group_std'].append(
                    np.std([info['length'] for info in group[split_idx:]]))
                if len(group[:split_idx]) > self.min_num_for_active_inspection:
                    pseudo_gt = np.mean([info['length'] for info in group[:split_idx]])
                    request = requests[np.abs(requests-pseudo_gt).argmin()]
                    if abs(request-pseudo_gt) < 1:
                        for j, info in enumerate(group[:split_idx]):
                            bias = abs(groups[gi][j]['length']-pseudo_gt)
                            error = abs(groups[gi][j]['length']-request)
                            if bias > pseudo_gt * self.tolerance_ratio:
                                groups[gi][j]['active_pass'] = False
                            if error > self.tolerance_bias:
                                groups[gi][j]['passive_pass'] = False
                if len(group[split_idx:]) > self.min_num_for_active_inspection:
                    pseudo_gt = np.mean([info['length'] for info in group[split_idx:]])
                    request = requests[np.abs(requests-pseudo_gt).argmin()]
                    if abs(request-pseudo_gt) < 1:
                        for j, info in enumerate(group[split_idx:]):
                            bias = abs(groups[gi][split_idx+j]['length']-pseudo_gt)
                            error = abs(groups[gi][split_idx+j]['length']-request)
                            if bias > pseudo_gt * self.tolerance_ratio:
                                groups[gi][split_idx+j]['active_pass'] = False
                            if error > self.tolerance_bias:
                                groups[gi][j]['passive_pass'] = False

            else:
                statistics['num_group'] += 1
                statistics['group_max'].append(np.max([info['length'] for info in group]))
                statistics['group_min'].append(np.min([info['length'] for info in group]))
                statistics['group_mean'].append(
                    np.mean([info['length'] for info in group]))
                statistics['group_std'].append(np.std([info['length'] for info in group]))
                if len(group) > self.min_num_for_active_inspection:
                    pseudo_gt = np.mean([info['length'] for info in group])
                    request = requests[np.abs(requests-pseudo_gt).argmin()]
                    if abs(request-pseudo_gt) < 1:
                        for i, info in enumerate(group):
                            bias = abs(groups[gi][i]['length']-pseudo_gt)
                            error = abs(groups[gi][i]['length']-request)
                            if bias > pseudo_gt * self.tolerance_ratio:
                                groups[gi][i]['active_pass'] = False
                            if error > self.tolerance_bias:
                                groups[gi][j]['passive_pass'] = False
        spacing_id = 0
        for group in groups.values():
            for info in group:
                points = info['points']
                mx, my = np.mean(points, axis=0, dtype=int)
                color = self.colors[1]
                if not info['passive_pass']:
                    color = (0, 0, 255)
                elif not info['active_pass']:
                    color = (0, 255, 255)

#                cv2.line(self.vis, points[0], points[1], color, 3)
#                cv2.circle(self.vis, points[0], 8, self.colors[0], -1)
#                cv2.circle(self.vis, points[1], 8, self.colors[0], -1)
#                text = '{:.2f}cm'.format(info['length'])
#                font = cv2.FONT_HERSHEY_SIMPLEX
#                (text_width, text_height) = cv2.getTextSize(
#                    text, font, fontScale=0.5, thickness=1)[0]
#                cv2.rectangle(self.vis, (mx-25, my-5), (mx +
#                              text_width-25, my-text_height-5), (0, 0, 0), -1)
#                cv2.putText(self.vis, text, (mx-25, my-5), font,
#                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
#                cv2.imwrite(os.path.join('ActiveInspection_'+name), self.vis)
                result['spacing']['S'+str(spacing_id).zfill(3)] = {
                    "intersection": info['id'],
                    "points": info['points'],
                    "length": info['length'],
                    "orientation": info['orientation'],
                    "active_pass": info['active_pass'],
                    "passive_pass": info['passive_pass']
                }
                spacing_id += 1

        result['statistics'] = statistics

        result['parameter'] = {
            'threshold': self.threshold,
            'max_pixel_dist': self.max_pixel_dist,
            'max_real_dist': self.max_real_dist,
            'min_num_for_active_inspection': self.min_num_for_active_inspection,
            'tolerance_ratio': self.tolerance_ratio,
            'tolerance_bias': self.tolerance_bias
        }
        return result
#        with open(os.path.join(self.output_dir, 'SPACING.json'), 'w') as ofile:
#            json.dump(result, ofile, indent=2)
#        ofile.close()
        

    def draw_spacing(self, bg, info):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for spacing in info['spacing']:
            pt1, pt2 = spacing['points']
            textx, texty = np.mean([pt1, pt2], axis=0, dtype=int)
            cv2.line(bg, pt1, pt2, self.colors[1], 2)
            cv2.circle(bg, pt1, 5, self.colors[0], -1)
            cv2.circle(bg, pt2, 5, self.colors[0], -1)
            text = '{:.2f}cm'.format(spacing['length'])
            (text_width, text_height) = cv2.getTextSize(
                text, font, fontScale=0.5, thickness=1)[0]
            cv2.rectangle(bg, (textx-25, texty-5), (textx+text_width -
                          25, texty-text_height-5), (0, 0, 0), -1)
            cv2.putText(bg, text, (textx-25, texty-5), font,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return bg


class HookExtensionModule:
    def __init__(self, threshold=0.8, device='cuda', model_path="./mrcnn_model/hookextension.pth", requests='./dataset/requests.xlsx'):
        cfg = get_cfg()
        cfg.thing_classes = ['arc', 'extension', 'end', 'line', '_background_']
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.DEVICE = device
        self.labels = ['Arc', 'Extension', 'End', 'Line', 'background']
        self.colors = [(26, 178, 165), (255, 51, 175),
                        (162, 124, 250), (171, 174, 249), None]
        self.model = DefaultPredictor(cfg)
        self.requests = requests
        self.threshold = threshold

    def predict_mask(self, color, depth, instances_angle=None):
        outputs = self.model(color)
        instances = outputs["instances"].to("cpu")
        
        if instances_angle is not None:
            angle_mask = torch.sum(instances_angle.pred_masks, axis=0)
            instances = instances[torch.sum(angle_mask * instances.pred_masks, axis=(1, 2)) > 0]
        
        mask = instance2mask(instances, self.colors)
        return mask, instances

    def predict_extension(self, instances, coord, name, max_pixel_dist=20, max_real_dist=2):
        classes = instances.pred_classes.cpu()
        masks = np.asarray(instances.pred_masks.cpu())
        points = [[] for i in range(5)]
        centroids = [[] for i in range(5)]
        for c, mask in zip(classes, masks):
            polygons = Mask(mask).polygons()
            if not polygons.points: continue
            pts = []
            for contour in polygons.points:
                pts.extend(contour)
            points[c].append(pts)
            M = cv2.moments(np.array(pts))
            centroids.append((int(M['m10'] / M["m00"]), int(M['m01'] / M["m00"])))

        info = []

        for extension_pts in points[1]:
            add = []
            minus = []
            for i in range(len(np.asarray(extension_pts).squeeze())):
                add.append((np.asarray(extension_pts).squeeze())[
                           i][0]+(np.asarray(extension_pts).squeeze())[i][1])
                minus.append((np.asarray(extension_pts).squeeze())[
                             i][0]-(np.asarray(extension_pts).squeeze())[i][1])
            add_max = [i for i in range(len(add)) if add[i] == add[np.argsort(add)[-1]]]
            if len(add_max) > 1:
                lst = []
                for idx in add_max:
                    lst.append((np.asarray(extension_pts).squeeze())[idx][0])
                X1 = (np.asarray(extension_pts).squeeze())[
                      add_max[np.asarray(lst).argmin()]][0]
                Y1 = (np.asarray(extension_pts).squeeze())[
                      add_max[np.asarray(lst).argmin()]][1]
            else:
                X1 = (np.asarray(extension_pts).squeeze())[np.argsort(add)[-1]][0]
                Y1 = (np.asarray(extension_pts).squeeze())[np.argsort(add)[-1]][1]

            add_min = [i for i in range(len(add)) if add[i] == add[np.argsort(add)[0]]]
            if len(add_min) > 1:
                lst = []
                for idx in add_min:
                    lst.append((np.asarray(extension_pts).squeeze())[idx][0])
                X2 = (np.asarray(extension_pts).squeeze())[
                      add_min[np.asarray(lst).argmax()]][0]
                Y2 = (np.asarray(extension_pts).squeeze())[
                      add_min[np.asarray(lst).argmax()]][1]
            else:
                X2 = (np.asarray(extension_pts).squeeze())[np.argsort(add)[0]][0]
                Y2 = (np.asarray(extension_pts).squeeze())[np.argsort(add)[0]][1]

            minus_max = [i for i in range(
                len(minus)) if minus[i] == minus[np.argsort(minus)[-1]]]
            if len(minus_max) > 1:
                lst = []
                for idx in minus_max:
                    lst.append((np.asarray(extension_pts).squeeze())[idx][0])
                X3 = (np.asarray(extension_pts).squeeze())[
                      minus_max[np.asarray(lst).argmin()]][0]
                Y3 = (np.asarray(extension_pts).squeeze())[
                      minus_max[np.asarray(lst).argmin()]][1]
            else:
                X3 = (np.asarray(extension_pts).squeeze())[np.argsort(minus)[-1]][0]
                Y3 = (np.asarray(extension_pts).squeeze())[np.argsort(minus)[-1]][1]

            minus_min = [i for i in range(
                len(minus)) if minus[i] == minus[np.argsort(minus)[0]]]
            if len(minus_min) > 1:
                lst = []
                for idx in minus_min:
                    lst.append((np.asarray(extension_pts).squeeze())[idx][0])
                X4 = (np.asarray(extension_pts).squeeze())[
                      minus_min[np.asarray(lst).argmax()]][0]
                Y4 = (np.asarray(extension_pts).squeeze())[
                      minus_min[np.asarray(lst).argmax()]][1]
            else:
                X4 = (np.asarray(extension_pts).squeeze())[np.argsort(minus)[0]][0]
                Y4 = (np.asarray(extension_pts).squeeze())[np.argsort(minus)[0]][1]

            P1 = coord[Y1, X1]
            P2 = coord[Y2, X2]
            P3 = coord[Y3, X3]
            P4 = coord[Y4, X4]
            length_box = [np.linalg.norm(P1-P2) * 100, np.linalg.norm(P1-P3) * 100, np.linalg.norm(
                P1-P4) * 100, np.linalg.norm(P2-P3) * 100, np.linalg.norm(P2-P4) * 100, np.linalg.norm(P3-P4) * 100]
            length = float(length_box[np.argsort(length_box)[2]])
            if np.argsort(length_box)[0] == 2 or np.argsort(length_box)[0] == 3:
                END1 = [int((X1+X4)/2), int((Y1+Y4)/2)]
                END2 = [int((X2+X3)/2), int((Y2+Y3)/2)]
            else:
                END1 = [int((X2+X4)/2), int((Y2+Y4)/2)]
                END2 = [int((X1+X3)/2), int((Y1+Y3)/2)]
            points = [END2, END1]
            if length > 0:
                info.append({
                    'points': points,
                    'length': length
                })
        result = self.inspect_info(info, name)
        return info, result

    def inspect_info(self, info, name):
        df = pd.read_excel(self.requests, usecols = [2])
        label=df.iloc[6, 0]  # 6th row & 0th col
        label=label.split('≧')
        label=float(label[1])/10  # Change from mm to cm
        
        rows = pd.read_excel(self.requests).values.tolist()
        inspect_item = '鋼筋彎鉤延伸長度'
        requests = np.array([float(row[row.index(inspect_item)+1].split('≧')[-1])/10 for row in rows if inspect_item in row])
        
        result={}
        json_obj={}
        json_obj["basename"]="{}.png".format(name)
        json_obj["parameter"]={"threshold": self.threshold}
        info_extension=info
        length = [info['length'] for info in info_extension]
        num_extension = len(info_extension)
        extension_max = max(length) if len(length) else 0
        extension_min = min(length) if len(length) else 0
        extension_mean = np.mean(length) if len(length) else 0
        extension_std = np.std(length) if len(length) else 0
        json_obj["statistics"]={
            "num_extension": len(info_extension),
            "extension_max": extension_max,
            "extension_min": extension_min,
            "extension_mean": extension_mean,
            "extension_std": extension_std
        }
        extension_json = {}
        for i, item in enumerate(info_extension):
            extention_name = "E{:0>3d}".format(i)

            info_json = {}
            info_json["points"] = [item['points'][0], item['points'][1]]
            info_json["length"] = item['length']
            info_json["active_pass"] = True
            info_json["passive_pass"] = bool(item['length'] >= requests[np.argmin(abs(item['length']-requests))])
            extension_json[extention_name] = info_json

        json_obj["extension"] = extension_json

        result.update(json_obj)

#        with open('./output/HOOKEXTENSION.json', 'w') as jsonFile:
#            json.dump(result, jsonFile, indent=2)
        return result

    def draw_extension(self, bg, info):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for extension in info:
            pt1, pt2 = extension['points']
            textx, texty = np.mean([pt1, pt2], axis=0, dtype=int)
            cv2.line(bg, tuple(pt1), tuple(pt2), self.colors[1], 2)
            cv2.circle(bg, tuple(pt1), 5, self.colors[0], -1)
            cv2.circle(bg, tuple(pt2), 5, self.colors[0], -1)
            text = '{:.2f}cm'.format(extension['length'])
            (text_width, text_height) = cv2.getTextSize(
                text, font, fontScale = 0.5, thickness = 1)[0]
            cv2.rectangle(bg, (textx-25, texty-5), (textx+text_width -
                          25, texty-text_height-5), (0, 0, 0), -1)
            cv2.putText(bg, text, (textx-25, texty-5), font,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return bg

class HookAngleModule:
    def __init__(self, threshold = 0.8, device = 'cuda', model_path = "./mrcnn_model/model_angle.pth", requests = './dataset/requests.xlsx'):
        cfg=get_cfg()
        cfg.thing_classes=['0', '90', '135', '180']
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS=4
        cfg.MODEL.ROI_HEADS.NUM_CLASSES=4
        cfg.MODEL.WEIGHTS=model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=threshold
        self.threshold=threshold
        cfg.MODEL.DEVICE=device
        self.labels=['None', '90 degree', '135 degree', '180 degree']
        self.colors=[None, (0, 200, 200), (200, 0, 0), (200, 200, 0)]
        self.model=DefaultPredictor(cfg)
        self.requests=requests

    def predict_mask(self, color, depth):
        outputs=self.model(color)
        instances=outputs["instances"].to("cpu")
        mask=instance2mask(instances, self.colors)
        return mask, instances

    def predict_angle(self, instances, name):
        info=[]
        classes=instances.pred_classes.cpu()
        boxes=instances.pred_boxes.tensor.numpy()
        for c, b in zip(classes, boxes):
            info.append({
                'points': [b[:2], b[2:]],
                'color': self.colors[c],
                'label': self.labels[c]
            })
        result=self.inspect_info(info, name)
        return info, result
    def inspect_info(self, info, name):
        rows = pd.read_excel(self.requests).values.tolist()
        inspect_item = '鋼筋彎鉤延伸長度'
        requests = np.array([row[row.index(inspect_item)+1].split('≧')[0]+' degree' for row in rows if inspect_item in row])
        
        result={}
        json_obj={}
        json_obj["basename"]="{}.png".format(name)
        json_obj["parameter"]={"threshold": self.threshold}
        info_angle=info
        json_obj["statistics"]={"num_angle": len(info_angle)}
        result.update(json_obj)
        for i, item in enumerate(info_angle):
            anglename="A{:0>3d}".format(i)
            info_json={}
            info_json["box"]=[item['points']
                [0].tolist(), item['points'][1].tolist()]
            info_json["angle"]=item['label']
            info_json["active_pass"]=(item['label'] != 'None')
            # 重新幫0 degree加回passive_pass，方便json轉成pdf
            info_json["passive_pass"]=bool(item['label'] in requests)
            json_obj[anglename]=info_json
            result.update(json_obj)

#        with open('HOOKANGLE.json', 'w') as jsonFile:
#            json.dump(result, jsonFile, indent=2)
        return result

    def draw_angle(self, bg, info):
        bg=np.array(bg, np.uint8)
        font=cv2.FONT_HERSHEY_SIMPLEX
        for angle in info:
            pt1, pt2=np.array(angle['points'], int)
            cv2.rectangle(bg, tuple(pt1), tuple(pt2), angle['color'], 1)
            cv2.putText(bg, angle['label'], tuple(pt1), font, 0.5, angle['color'], 1, cv2.LINE_AA)
        return bg

class OverlapModule:
    def __init__(self, threshold=0.8, device='cuda', model_path="./mrcnn_model/model_overlap.pth", requests='./dataset/requests.xlsx'):
        cfg=get_cfg()
        cfg.thing_classes=['edge', 'overlap']
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS= 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES= 2
        cfg.MODEL.WEIGHTS= model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST= threshold
        cfg.MODEL.DEVICE= device
        self.labels= ['Edge', 'Overlap']
        self.colors = [(59, 165,31), (164,210,152)]
        self.model= DefaultPredictor(cfg)
        self.requests= requests
        self.threshold = threshold
        self.tolerance_bias = 6

    def predict_mask(self, color, depth):
        outputs= self.model(color)
        instances= outputs["instances"].to("cpu")
        mask= instance2mask(instances, self.colors)
        return mask, instances

    def predict_overlap(self, instances, coord, f):
        classes= instances.pred_classes.cpu()
        masks= np.asarray(instances.pred_masks.cpu())
        points= [[] for label in self.labels]
        cnt= [[] for label in self.labels]
        for c, mask in zip(classes, masks):
            polygons= Mask(mask).polygons()
            if not polygons.points:
                continue
            pts= []
            for contour in polygons.points:
                pts.extend(contour)
            points[c].append(pts)
            M= cv2.moments(np.array(pts))
            cnt[c].append([int(M['m10'] / M["m00"]), int(M['m01'] / M["m00"])])
        edge_dic= dict()
        for i in range(len(cnt[0])):
            edge_dic['E'+str(i).rjust(3, '0')]=cnt[0][i]
        info= []
        overlapinfo=[]
        position_left= []
        position_right= []
        for overlap_pts in points[1]:
            add= []
            minus= []
            for i in range(len(np.asarray(overlap_pts).squeeze())):
                add.append((np.asarray(overlap_pts).squeeze())[
                           i][0]+(np.asarray(overlap_pts).squeeze())[i][1])
                minus.append((np.asarray(overlap_pts).squeeze())[
                             i][0]-(np.asarray(overlap_pts).squeeze())[i][1])
            add_max= [i for i in range(len(add)) if add[i] == add[np.argsort(add)[-1]]]
            if len(add_max) >1:
                lst= []
                for idx in add_max:
                    lst.append((np.asarray(overlap_pts).squeeze())[idx][0])
                X1= (np.asarray(overlap_pts).squeeze())[add_max[np.asarray(lst).argmin()]][0]
                Y1= (np.asarray(overlap_pts).squeeze())[add_max[np.asarray(lst).argmin()]][1]
            else:
                X1= (np.asarray(overlap_pts).squeeze())[np.argsort(add)[-1]][0]
                Y1= (np.asarray(overlap_pts).squeeze())[np.argsort(add)[-1]][1]

            add_min= [i for i in range(len(add)) if add[i] == add[np.argsort(add)[0]]]
            if len(add_min) >1:
                lst= []
                for idx in add_min:
                    lst.append((np.asarray(overlap_pts).squeeze())[idx][0])
                X2= (np.asarray(overlap_pts).squeeze())[add_min[np.asarray(lst).argmax()]][0]
                Y2= (np.asarray(overlap_pts).squeeze())[add_min[np.asarray(lst).argmax()]][1]
            else:
                X2= (np.asarray(overlap_pts).squeeze())[np.argsort(add)[0]][0]
                Y2= (np.asarray(overlap_pts).squeeze())[np.argsort(add)[0]][1]

            minus_max= [i for i in range(len(minus)) if minus[i] == minus[np.argsort(minus)[-1]]]
            if len(minus_max) >1:
                lst= []
                for idx in minus_max:
                    lst.append((np.asarray(overlap_pts).squeeze())[idx][0])
                X3= (np.asarray(overlap_pts).squeeze())[minus_max[np.asarray(lst).argmin()]][0]
                Y3= (np.asarray(overlap_pts).squeeze())[minus_max[np.asarray(lst).argmin()]][1]
            else:
                X3= (np.asarray(overlap_pts).squeeze())[np.argsort(minus)[-1]][0]
                Y3= (np.asarray(overlap_pts).squeeze())[np.argsort(minus)[-1]][1]

            minus_min= [i for i in range(len(minus)) if minus[i] == minus[np.argsort(minus)[0]]]
            if len(minus_min) >1:
                lst= []
                for idx in minus_min:
                    lst.append((np.asarray(overlap_pts).squeeze())[idx][0])
                X4= (np.asarray(overlap_pts).squeeze())[minus_min[np.asarray(lst).argmax()]][0]
                Y4= (np.asarray(overlap_pts).squeeze())[minus_min[np.asarray(lst).argmax()]][1]
            else:
                X4= (np.asarray(overlap_pts).squeeze())[np.argsort(minus)[0]][0]
                Y4= (np.asarray(overlap_pts).squeeze())[np.argsort(minus)[0]][1]

            P1 = coord[Y1, X1]
            P2 = coord[Y2, X2]
            P3 = coord[Y3, X3]
            P4 = coord[Y4, X4]
            length_box= [np.linalg.norm(P1-P2) * 100, np.linalg.norm(P1-P3) * 100, np.linalg.norm(P1-P4) * 100, np.linalg.norm(P2-P3) * 100, np.linalg.norm(P2-P4) * 100, np.linalg.norm(P3-P4) * 100]
            length= int(length_box[np.argsort(length_box)[2]])
            if np.argsort(length_box)[0] == 2 or np.argsort(length_box)[0] == 3:
                END1 = [int((X1+X4)/2), int((Y1+Y4)/2)]
                END2 = [int((X2+X3)/2), int((Y2+Y3)/2)]
            else:
                END1 = [int((X2+X4)/2), int((Y2+Y4)/2)]
                END2 = [int((X1+X3)/2), int((Y1+Y3)/2)]
            points= [END2, END1]
            info.append({
                'points': points,
                'length': length
            })
            cond = 'incomplete'
            e2 = END2
            position_left.append(e2)
            e1 = END1
            position_right.append(e1)
            e=[]
            if len(cnt[0])>0:
                dst1 = [math.hypot((END1[0]-cntpt[0]),(END1[1]-cntpt[1])) for cntpt in cnt[0]]
                dst2 = [math.hypot((END2[0]-cntpt[0]),(END2[1]-cntpt[1])) for cntpt in cnt[0]]
                if min(dst1)<=50 and min(dst2)<=50:
                    cond = 'complete'
                    E1X = cnt[0][np.asarray(dst1).argmin()][0]
                    E1Y = cnt[0][np.asarray(dst1).argmin()][1]
                    E2X = cnt[0][np.asarray(dst2).argmin()][0]
                    E2Y = cnt[0][np.asarray(dst2).argmin()][1]
                    e1 = [E1X,E1Y]
                    position_right.append(e1)
                    e2 = [E2X,E2Y]
                    position_left.append(e2)
                    e=(['E'+str(np.asarray(dst1).argmin()).rjust(3,'0'),\
                             'E'+str(np.asarray(dst2).argmin()).rjust(3,'0')])
                elif min(dst1)<=50 and min(dst2)>50: #and min(dst2)>15
                    E1X = cnt[0][np.asarray(dst1).argmin()][0]
                    E1Y = cnt[0][np.asarray(dst1).argmin()][1]
                    e1 = [E1X,E1Y]
                    position_right.append(e1)
                    e2 = END2
                    position_left.append(e2)
                    e=(['E'+str(np.asarray(dst1).argmin()).rjust(3,'0')])
                elif min(dst2)<=50 and min(dst1)>50: #and min(dst2)<=15
                    E2X = cnt[0][np.asarray(dst2).argmin()][0]
                    E2Y = cnt[0][np.asarray(dst2).argmin()][1]
                    e2 = [E2X,E2Y]
                    position_left.append(e2)
                    e1 = END1
                    position_right.append(e1)
                    e=(['E'+str(np.asarray(dst2).argmin()).rjust(3,'0')])
                                    
            overlapinfo.append({'len':length,'s':cond,\
                                'edge':e,'points':[np.array(END1),np.array(END2)],'mid':(np.array(END1)+np.array(END2))/2})
            
        overlap_dic = dict()
        rows = pd.read_excel(self.requests).values.tolist()
        inspect_item = '搭接長度'
        requests = np.array([float(row[row.index(inspect_item)+1]) for row in rows if inspect_item in row])

        n=0
        for i in range(len(overlapinfo)):
            for j in range(i+1,len(overlapinfo)):
                if np.linalg.norm(overlapinfo[i].get('mid')-overlapinfo[j].get('mid'))<=50:
                    p=[((overlapinfo[i].get('points')[0]+overlapinfo[j].get('points')[0])/2).tolist(),((overlapinfo[i].get('points')[1]+overlapinfo[j].get('points')[1])/2).tolist()]
                    
                    if (p[0][0]-p[1][0])>=0:
                        vector = complex((p[0][0]-p[1][0]),(p[0][1]-p[1][1]))
                    else:
                        vector = complex(-(p[0][0]-p[1][0]),-(p[0][1]-p[1][1]))
                    length = round((overlapinfo[i].get('len')+overlapinfo[j].get('len'))/2,2)
                    
                    overlap_dic['O'+str(n).rjust(3,'0')]=\
                    dict(edge=overlapinfo[i].get('edge'),
                         points=p,
                         length=length,
                         orientation=round(np.angle(vector,deg=True),2),
                         status=overlapinfo[i].get('s'),
                         active_pass=True,
                         passive_pass=bool(min(abs(request-length) for request in requests) < self.tolerance_bias)
                         )
        result = dict(basename=f,
                    statistics=dict(num_overlap=len(overlapinfo)),
                    parameter=dict(threshold=self.threshold, tolerance_bias=self.tolerance_bias),
                    edge=edge_dic,
                    overlap=overlap_dic
                    )
#        with open('./output/Overlap.json','w') as f:
#            json.dump(result,f,indent=2)
#            f.write('\n')
        return overlapinfo, result
    
    def draw_overlap(self, bg, overlapinfo):
        bg = np.array(bg, np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(overlapinfo)):
            for j in range(i+1,len(overlapinfo)):
                if np.linalg.norm(overlapinfo[i].get('mid')-overlapinfo[j].get('mid'))<=50:
                    p=[((overlapinfo[i].get('points')[0]+overlapinfo[j].get('points')[0])/2).tolist(),((overlapinfo[i].get('points')[1]+overlapinfo[j].get('points')[1])/2).tolist()]

                    text = 'Overlap={:.2f}cm'.format(round((overlapinfo[i].get('len')+overlapinfo[j].get('len'))/2,2))
                    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=0.5, thickness=1)[0]
                    # cv2.line(bgd, tuple(END1), tuple(END2), [0, 0, 0], 5)
                    MX = int((p[0][0]+p[1][0])/2)
                    MY = int((p[0][1]+p[1][1])/2)
                    cv2.rectangle(bg, (MX+10, MY+10), (MX+10+text_width, MY+10-text_height), self.colors[0], -1)
                    cv2.putText(bg, text, (MX+10, MY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[1], 1, cv2.LINE_AA)
        return bg

class Modules(object):
    def __init__(self, device='cuda', clear=False):
        self.requests_path = "dataset/requests.xlsx"
        self.Spacing = SpacingModule(device=device, threshold=0.5, model_path="./mrcnn_model/model_spacing_v1.pth", requests=self.requests_path)
        
        self.HookAngle = HookAngleModule(device=device, threshold=0.5, model_path="./mrcnn_model/model_angle.pth", requests=self.requests_path)
        
        self.Overlap = OverlapModule(device=device, threshold=0.5, model_path="./mrcnn_model/model_overlap_v2.pth", requests=self.requests_path)
        
        self.HookExtension = HookExtensionModule(device=device, threshold=0.5, model_path="./mrcnn_model/model_extension.pth", requests=self.requests_path)
        
        self.labels = self.Spacing.labels + self.HookAngle.labels + self.Overlap.labels + self.HookExtension.labels
        self.colors = self.Spacing.colors + self.HookAngle.colors + self.Overlap.colors + self.HookExtension.colors
        
        self.preload([self.Spacing, self.HookAngle, self.Overlap, self.HookExtension], times=5)
        
        self.viewer = PicViewer(cover=cv2.imread('dataset/cover.png'))
        
        threading.Thread(target=self.viewer.run).start()
        if clear and os.path.isdir('predictions'):
            shutil.rmtree('predictions')
        # self.preload([self.Spacing, self.HookAngle], times=10)
    
    def preload(self, modules, times=4):        
        color = np.ones((720, 1280, 3), dtype=np.uint8)
        depth = np.ones((720, 1280, 3), dtype=np.uint8)
        for t in range(times):
            for n, module in enumerate(modules):
                module.predict_mask(color, depth)
            print('Preloading... ({:^3}/{:^3})'.format(t+1, times), end='\r')
    
    def predict_mask(self, color, depth):
        mask_spacing, instances_spacing = self.Spacing.predict_mask(color, depth)
        mask_angle, instances_angle = self.HookAngle.predict_mask(color, depth)
        mask_overlap, instances_overlap = self.Overlap.predict_mask(color, depth)
        mask_extension, instances_extension = self.HookExtension.predict_mask(color, depth)
        
        mask = draw_masks(color, [mask_overlap, mask_spacing, mask_angle, mask_extension])
        # mask = draw_masks(color, [mask_spacing, mask_angle])
        
        self.viewer.update_mask(mask)
        return mask
    
    def predict(self, color, depth, coord):
        os.makedirs('predictions', exist_ok=True)
        mask_spacing, instances_spacing = self.Spacing.predict_mask(color, depth)
        mask_angle, instances_angle = self.HookAngle.predict_mask(color, depth)
        mask_overlap, instances_overlap = self.Overlap.predict_mask(color, depth)
        mask_extension, instances_extension = self.HookExtension.predict_mask(color, depth, instances_angle=instances_angle)
        
        mask = draw_masks(color, [mask_overlap, mask_spacing, mask_angle, mask_extension])
        
        name = self.viewer.update_mask(mask)
        info_spacing, result_spacing = self.Spacing.predict_spacing(instances_spacing, coord, name)
        info_angle, result_angle = self.HookAngle.predict_angle(instances_angle, name)
        info_overlap, result_overlap = self.Overlap.predict_overlap(instances_overlap, coord, name)
        info_extension, result_extension = self.HookExtension.predict_extension(instances_extension, coord, name)
        parameter = self.HookAngle.draw_angle(color, info_angle)
        parameter = self.Spacing.draw_spacing(parameter, info_spacing)
        parameter = self.Overlap.draw_overlap(parameter, info_overlap)
        parameter = self.HookExtension.draw_extension(parameter, info_extension)
        self.viewer.update_parameter(parameter, name)
        
        img = Image.fromarray(color.astype("uint8"))
        bytes = io.BytesIO()
        img.save(bytes, "PNG")
        data = bytes.getvalue()
        encData = codecs.encode(data, 'base64').decode()
        imageData = encData.replace('\n', '')
        t = os.path.splitext(name)[0]
        report = {}
        rows = pd.read_excel(self.requests_path, header=None, keep_default_na=False).values.tolist()
        for row in rows:
            if row[0]:
                subtitle = row[0]
                report[subtitle] = {row[1]: row[2]}
            else:
                report[subtitle][row[1]] = row[2]
        report['ImagePath'] = name
        report['Time'] = t
        report['Spacing'] = result_spacing
        report['HookAngle'] = result_angle
        report['HookExtension'] = result_extension
        report['Overlap'] = result_overlap
        report['ImageData'] = None
        f = open('predictions/{}_info.json'.format(t),'w')
        json.dump(report, f, indent=2)
        f.close()
        writepdf('predictions/{}_report.pdf'.format(t), report)
        os.system('pkill chrome')
        webbrowser.open(r'predictions/{}_report.pdf'.format(t))
        return mask, parameter
        
if __name__ == '__main__':
    HookAngle = HookAngleModule(threshold=0.8, device='cuda')
    Spacing = SpacingModule(threshold=0.8, device='cuda')
    Overlap = OverlapModule(threshold=0.5, device='cuda')
    HookExtension = HookExtensionModule(threshold=0.7, device='cuda')
