import warnings
warnings.filterwarnings("ignore")

# import some common libraries
import sys, os, glob, re, time, distutils.core

# Project root is two levels up from this file (code/lib/ -> code/ -> root)
_PROJ_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_PROJECTS  = os.path.join(_PROJ_ROOT, 'projects')
import torch, detectron2
import numpy as np
import json, cv2, random
import matplotlib.pyplot as plt
import csv

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode, _PanopticPrediction
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

class Detector:
    def __init__(self, model, model_type):
        self.model = model
        self.cfg = get_cfg()
        self.model_type = model_type

        #load model config and pretrained model
        if self.model == 'COCO':
           if model_type == 'OD': #object detection
              self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
              self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
              self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
              self.classes = [9,11] # 0-->person, 2-->car, 9-->traffic light, 10-->fire hydrant, 11-->stop sign, 13-->bench
           elif model_type == 'IS': #instance segmentation
               self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
               self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
               self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
               self.classes = [9,10,11,13]
           elif model_type == 'P': #panoptic
               self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
               self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
               self.stuff_classes = [] #21-->road, 28-->house, 29-->light, 37-->tree, 40-->sky, 44-->pavement, 45-->mountain, 50-->building, 52-->wall
               self.thing_classes = [0,1,2,3,5,6,7] # 0-->person, 1--> bicycle, 2-->car, 3-->motorcycle, 5-->bus, 6-->train, 7-->truck, 9-->traffic light, 10-->fire hydrant, 11-->stop sign, 13-->bench
               self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
           else : print('Invalid model type. Valid model type options : OD, IS, P')

        elif self.model == 'MaskFormer': #not working!!!
            if model_type == 'P': #panoptic
               sys.path.append(_PROJECTS)
               from projects.Mask2Former.mask2former.config import add_maskformer2_config # Import MaskFormer configs
               add_maskformer2_config(self.cfg)
               self.cfg.merge_from_file(os.path.join(_PROJECTS, 'MaskFormer', 'panoptic', 'maskformer2_R50_bs16_50ep.yaml'))
               self.cfg.MODEL.WEIGHTS = os.path.join(_PROJECTS, 'MaskFormer', 'panoptic', 'model_final_94dc52.pkl')
               self.stuff_classes = [40] #21-->road, 28-->house, 29-->light, 37-->tree, 40-->sky, 44-->pavement, 45-->mountain, 50-->building, 52-->wall
               self.thing_classes = [0,2] # 0-->person, 1--> bicycle, 2-->car, 3-->motorcycle, 5-->bus, 6-->train, 7-->truck, 9-->traffic light, 10-->fire hydrant, 11-->stop sign, 13-->bench
               self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 1
            else : print('Invalid model type. Valid model type options : P')

        elif self.model == 'Cityscapes':
            if model_type == 'SS': #semantic segmentation
                sys.path.append(_PROJECTS)
                from projects.DeepLab.deeplab.config import add_deeplab_config
                add_deeplab_config(self.cfg)
                self.cfg.merge_from_file(os.path.join(_PROJECTS, 'Cityscapes', 'sem_seg', 'deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml'))
                self.cfg.MODEL.WEIGHTS = os.path.join(_PROJECTS, 'Cityscapes', 'sem_seg', 'model_final_a8a355.pkl')
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
                self.cfg.INPUT.CROP.ENABLED = False
            elif model_type == 'P':
                sys.path.append(_PROJECTS)
                from projects.Panoptic_DeepLab.panoptic_deeplab.config import add_panoptic_deeplab_config
                add_panoptic_deeplab_config(self.cfg)
                self.cfg.merge_from_file(os.path.join(_PROJECTS, 'Cityscapes', 'panoptic', 'panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml'))
                self.cfg.MODEL.WEIGHTS = os.path.join(_PROJECTS, 'Cityscapes', 'panoptic', 'model_final_bd324a.pkl')
                self.stuff_classes = [5] #0-->road, 5-->pole, 6-->traffic light, 7-->traffic sign
                self.thing_classes = [5] #0-->road, 5-->pole, 6-->traffic light, 7-->traffic sign
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
            else : print('Invalid model type. Valid model type options : SS, P')

        elif self.model == 'Crosswalk':
            if model_type == 'OD':
                self.cfg.merge_from_file(os.path.join(_PROJECTS, 'Crosswalk', 'output', 'config.yaml'))
                self.cfg.MODEL.WEIGHTS = os.path.join(_PROJECTS, 'Crosswalk', 'output', 'model_final.pth')
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
                self.classes = ['','Crosswalk']
            else : print('Invalid model type. Valid model type option : OD')
        
        elif self.model == 'Traffic_Sign':
            if model_type == 'OD':
                self.cfg.merge_from_file(os.path.join(_PROJECTS, 'Traffic_Sign', 'output', 'config.yaml'))
                self.cfg.MODEL.WEIGHTS = os.path.join(_PROJECTS, 'Traffic_Sign', 'output', 'model_final.pth')
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55
                self.classes = ["","Attention","Bend_to_left","Bend_to_right","Crosswalk","Fork_road","Give_way","Narrow_road","No_entry","No_left_turn","No_right_turn","No_u_turn","Roundabout_mandatory","Speed_limit_100KM","Speed_limit_110KM","Speed_limit_120KM","Speed_limit_20KM","Speed_limit_30KM","Speed_limit_40KM","Speed_limit_50KM","Speed_limit_60KM","Speed_limit_70KM","Speed_limit_80KM","Speed_limit_90KM","Stop"]
            else : print('Invalid model type. Valid model type option : OD')

        elif self.model == 'Safety_Cones':
            if model_type == 'OD':
                self.cfg.merge_from_file(os.path.join(_PROJECTS, 'Safety_Cones', 'output', 'config.yaml'))
                self.cfg.MODEL.WEIGHTS = os.path.join(_PROJECTS, 'Safety_Cones', 'output', 'model_final.pth')
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
                self.classes = ['','Safety_Cone']
            else : print('Invalid model type. Valid model type option : OD')
        
        else :  print('Invalid model. Valid model options : COCO, Cityscapes, Crosswalk, Traffic_Sign, Safety_Cones') 

        self.cfg.MODEL.DEVICE = 'cuda' # cpu or cuda
        self.cfg.freeze() # Κλειδώνει το CfgNode (config) και όλα τα παράγωγα αυτού 
        self.predictor= DefaultPredictor(self.cfg)
    
    def onImage(self, im):

        im_shape = im.shape
        
        if self.model_type == 'P':

            #Add padding if needed :

            if im_shape[0] != im_shape[1] :
                new_im_shape = max(im_shape[0],im_shape[1])

                b1 = 0 + int((new_im_shape - im_shape[0])/2)
                b2 = 0 + int((new_im_shape - im_shape[0])/2)
                b3 = 0 + int((new_im_shape - im_shape[1])/2)
                b4 = 0 + int((new_im_shape - im_shape[1])/2)

                im = cv2.copyMakeBorder(im, b1, b2, b3, b4, cv2.BORDER_CONSTANT,value=0)

            panoptic_seg, segments_info = self.predictor(im)["panoptic_seg"]
            
            if segments_info is not None: 
                 i = 0 
                 my_segments_info = []
                 while i < len(segments_info):
                      if segments_info[i]['isthing']==False and segments_info[i]["category_id"] in self.stuff_classes :
                          my_segments_info.append(segments_info[i])
                      elif segments_info[i]['isthing']==True and segments_info[i]["category_id"] in self.thing_classes :
                          my_segments_info.append(segments_info[i])
                      i = i+1
            else : 
                 metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
                 label_divisor = metadata.label_divisor
                 segments_info = []
                 for panoptic_label in np.unique(panoptic_seg.numpy()):
                     if panoptic_label == -1:
                      # VOID region.
                        continue
                     pred_class = panoptic_label // label_divisor
                     isthing = pred_class in metadata.thing_dataset_id_to_contiguous_id.values()
                     segments_info.append(
                     {
                        "id": int(panoptic_label),
                        "category_id": int(pred_class),
                        "isthing": bool(isthing),
                     }
                     )

                     i = 0 
                     my_segments_info = []
                     while i < len(segments_info):
                         if segments_info[i]['isthing']==False and segments_info[i]["category_id"] in self.stuff_classes :
                             my_segments_info.append(segments_info[i])
                         elif segments_info[i]['isthing']==True and segments_info[i]["category_id"] in self.thing_classes :
                              my_segments_info.append(segments_info[i])
                         i = i+1
            
            outputs = [{'panoptic_seg':panoptic_seg, 'my_segments_info':my_segments_info}]
            
        elif self.model_type == 'SS':
            outputs = torch.max(self.predictor(im)["sem_seg"], dim=0)[1]

        else :
            outputs = self.predictor(im)
            
        return outputs
    
    def output(self, im, outputs, name, directory, mode="mask"):

        # ============================================================
        # BLUR MODE  —  applies to ALL model types
        # Blurs detected regions and returns the blurred image.
        # The caller is responsible for saving (and rotating for Mode B).
        # ============================================================
        if mode == "blur":
            im_blur = im.copy()
            blurred = cv2.GaussianBlur(im, (31, 31), 0)

            if self.model_type == 'OD':
                instances = outputs["instances"].to("cpu")
                if self.model == 'COCO':
                    keep      = np.isin(instances.pred_classes.numpy(), self.classes)
                    instances = instances[keep]
                for box in instances.pred_boxes.tensor.numpy().astype(int):
                    x1 = max(box[0], 0);  y1 = max(box[1], 0)
                    im_blur[y1:box[3], x1:box[2]] = blurred[y1:box[3], x1:box[2]]

            elif self.model_type == 'IS':
                instances = outputs["instances"].to("cpu")
                if self.model == 'COCO':
                    keep      = np.isin(instances.pred_classes.numpy(), self.classes)
                    instances = instances[keep]
                if instances.has("pred_masks") and len(instances) > 0:
                    region = instances.pred_masks.numpy().any(axis=0)
                    im_blur[region] = blurred[region]

            elif self.model_type == 'P':
                h, w = im.shape[:2]
                for row in outputs:
                    panoptic_seg     = row['panoptic_seg']
                    my_segments_info = row['my_segments_info']
                pad_v = int((max(h, w) - h) / 2)
                pad_u = int((max(h, w) - w) / 2)
                if panoptic_seg.shape[0] != h or panoptic_seg.shape[1] != w:
                    panoptic_seg = panoptic_seg[pad_v:pad_v+h, pad_u:pad_u+w]
                region = torch.zeros(h, w, dtype=torch.uint8)
                for seg_info in my_segments_info:
                    region[panoptic_seg == seg_info['id']] = 1
                region_np = region.cpu().numpy().astype(bool)
                im_blur[region_np] = blurred[region_np]

            elif self.model_type == 'SS':
                class_map = outputs.cpu().numpy()
                im_blur[class_map > 0] = blurred[class_map > 0]

            return im_blur   # caller saves

        # ============================================================
        # MASK MODE (default)  —  save binary masks; return visualisation
        # ============================================================
        im_shape     = im.shape
        new_im_shape = max(im_shape[0], im_shape[1])

        if self.model_type == 'OD':
            info = {'instances': outputs}
            csv_name = os.path.join(directory, f"{name}_{self.model}_OD.csv")
            with open(csv_name, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=info.keys())
                writer.writeheader()
                writer.writerow(info)

        if self.model_type == 'P':
            for row in outputs:
                panoptic_seg     = row['panoptic_seg']
                my_segments_info = row['my_segments_info']
            b1 = int((new_im_shape - im_shape[0]) / 2)
            b3 = int((new_im_shape - im_shape[1]) / 2)
            if panoptic_seg.shape != im_shape:
                panoptic_seg = panoptic_seg[b1:b1+im_shape[0], b3:b3+im_shape[1]]
            for info in my_segments_info:
                seg_id  = info['id']
                mask    = (panoptic_seg == seg_id).to(torch.uint8)
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                if 'instance_id' not in info:
                    info['instance_id'] = '0'
                out_name = f"{name}_{self.model}_{info['category_id']}_{info['instance_id']}.jpg"
                cv2.imwrite(os.path.join(directory, out_name), mask_np)
            v   = Visualizer(im[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
            out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), my_segments_info,
                                                  area_threshold=None, alpha=0.7)

        elif self.model_type == 'SS':
            v   = Visualizer(im[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
            out = v.draw_sem_seg(outputs.to("cpu"), area_threshold=None, alpha=0.8)

        elif self.model in ['Crosswalk', 'Traffic_Sign', 'Safety_Cones']:
            MetadataCatalog.get("my_dataset_train").thing_classes = self.classes
            metadata = MetadataCatalog.get("my_dataset_train")
            v   = Visualizer(im[:,:,::-1], metadata, scale=1)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            MetadataCatalog.remove("my_dataset_train")

        else:
            v = Visualizer(im[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
            for i in self.classes:
                out = v.draw_instance_predictions(
                    outputs["instances"][outputs["instances"].pred_classes == i].to("cpu"))

        return out.get_image()[:,:,::-1]

    def get_od_centroids(self, outputs, image_name):
        """
        Extract bounding-box centres from OD predictions.

        Returns a list of dicts:
            point_name  : "<image_name>_<class_name>_<i>"
            image_name  : image_name (as passed in)
            u           : centre x in pixels
            v           : centre y in pixels
        """
        instances = outputs['instances'].to('cpu')
        boxes     = instances.pred_boxes.tensor.numpy()   # (N, 4) x1y1x2y2
        classes   = instances.pred_classes.numpy()
        H_img, W_img = instances.image_size

        # For COCO/IS models filter to only the requested class IDs
        if self.model == 'COCO' and len(self.classes) > 0 and isinstance(self.classes[0], int):
            keep  = np.isin(classes, self.classes)
            boxes   = boxes[keep]
            classes = classes[keep]

        # Build class-name lookup
        if self.model in ['Crosswalk', 'Traffic_Sign', 'Safety_Cones']:
            class_names = self.classes          # list with placeholder at index 0
        else:
            from detectron2.data import MetadataCatalog
            meta = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
            class_names = meta.thing_classes   # list indexed by class id

        centroids = []
        for i, (box, cls) in enumerate(zip(boxes, classes)):
            u = float((box[0] + box[2]) / 2.0)
            v = float((box[1] + box[3]) / 2.0)
            try:
                cls_name = class_names[int(cls)]
            except (IndexError, TypeError):
                cls_name = str(int(cls))
            centroids.append({
                'point_name':  f"{image_name}_{cls_name}_{i}",
                'image_name':  image_name,
                'cls':         cls_name,
                'u':           u,
                'v':           v,
                'bbox_norm_w': float(box[2] - box[0]) / W_img,
                'bbox_norm_h': float(box[3] - box[1]) / H_img,
            })

        return centroids


# ============================================================
# PIPELINE FUNCTION
# ============================================================

def run_detection(image_folder, models, mode,
                  cal_file=None,
                  output_masks=None,
                  output_coords=None,
                  output_mode='mask'):
    """
    Run the detection pipeline on a folder of images.

    Parameters
    ----------
    image_folder  : str   Folder containing input images.
    models        : list  [{'model': ..., 'model_type': ...}, ...]
    mode          : str   'A' = panoramic (no rotation)
                          'B' = raw Ladybug (rotate 90° CW, then unrotate)
    cal_file      : str   Path to .cal calibration file (required for mode B).
    output_masks  : str   Destination for mask images  (default: output/masks/).
    output_coords : str   Destination for CSV files    (default: output/coords/).
    output_mode   : str   'mask' (default) — save binary masks / centroid CSVs
                                             → produces image_coords.csv
                          'blur'           — Gaussian-blur detected regions and
                                             save the blurred images; no coords
                                             produced.

    Returns
    -------
    str   'mask' mode: absolute path to image_coords.csv.
          'blur' mode: absolute path to the blurred-images output folder.
    """
    from Centroid import compute_centroids

    masks_dir  = output_masks  or os.path.join(_PROJ_ROOT, 'output', 'masks')
    coords_dir = output_coords or os.path.join(_PROJ_ROOT, 'output', 'coords')
    os.makedirs(masks_dir,  exist_ok=True)
    os.makedirs(coords_dir, exist_ok=True)

    transformer = None
    if mode == 'B':
        assert cal_file, "cal_file is required for mode='B'"
        from raw_to_panorama import RawLadybugTransformer
        transformer = RawLadybugTransformer(cal_file)

    has_seg = any(m['model_type'] in ('P', 'SS') for m in models)
    has_od  = any(m['model_type'] == 'OD'        for m in models)

    detectors = [Detector(model=k['model'], model_type=k['model_type'])
                 for k in models]

    images  = sorted(glob.glob(os.path.join(image_folder, '*.jpg')) +
                     glob.glob(os.path.join(image_folder, '*.png')))
    od_rows = []

    blur_dir = os.path.join(_PROJ_ROOT, 'output', 'blurred')
    if output_mode == 'blur':
        os.makedirs(blur_dir, exist_ok=True)

    # ---- detection loop ----
    print("=" * 60)
    print(f"Detection  (Mode {mode}, output={output_mode})")
    print("=" * 60)

    for imagePath in images:
        name1 = os.path.basename(imagePath)

        # skip-check uses blur_dir for blur mode, masks_dir for mask mode
        check_dir = blur_dir if output_mode == 'blur' else masks_dir
        if glob.glob(os.path.join(check_dir, f"{name1}*")):
            print(f"  {name1}  [already processed, skipping]")
            continue

        img = cv2.imread(imagePath)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        img_det = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if mode == 'B' else img

        t0 = time.time()

        if output_mode == 'blur':
            # Apply all detectors in sequence; each one blurs on top of the previous.
            # Detection inference always runs on the original (unblurred) img_det.
            # Blurred image is saved in detection space (no back-rotation for Mode B).
            im_blurred = img_det.copy()
            for detector in detectors:
                out        = detector.onImage(img_det)
                im_blurred = detector.output(im_blurred, out, name1, blur_dir, mode="blur")
            cv2.imwrite(os.path.join(blur_dir, name1), im_blurred)

        else:
            # mask mode: extract masks / centroids
            for detector in detectors:
                out = detector.onImage(img_det)
                if detector.model_type == 'OD':
                    centroids = detector.get_od_centroids(out, name1)
                    if mode == 'B':
                        m = re.search(r'Cam(\d+)', name1)
                        cam_id = int(m.group(1)) if m else -1
                        iop    = transformer.iop.get(cam_id, {})
                        fx_norm = iop.get('fx_norm', 0.0)
                        for c in centroids:
                            # angular_w in radians — computed in raw camera frame
                            # before rotation swap, using pinhole focal length
                            c['angular_w'] = (c['bbox_norm_w'] / fx_norm
                                              if fx_norm > 0 else 0.0)
                            u_rot, v_rot = c['u'], c['v']
                            c['u']      = v_rot
                            c['v']      = orig_h - 1 - u_rot
                            c['cam_id'] = cam_id
                            c['img_w']  = orig_w
                            c['img_h']  = orig_h
                    od_rows.extend(centroids)
                else:
                    detector.output(img_det, out, name1, masks_dir, mode="mask")

        print(f"  {name1}  [{time.time()-t0:.2f}s]")

    # ---- blur mode: done here ----
    if output_mode == 'blur':
        print(f"\nBlurred images -> {blur_dir}")
        return blur_dir

    # ---- mask mode: post-processing ----
    coords_csv     = os.path.join(coords_dir, 'image_coords.csv')
    raw_coords_csv = os.path.join(coords_dir, 'raw_coords.csv')

    if has_seg:
        if mode == 'A':
            compute_centroids(masks_dir, coords_csv, mode='A')
        else:
            compute_centroids(masks_dir, raw_coords_csv, mode='B')
            transformer.transform_csv(raw_coords_csv, coords_csv)

    if has_od and od_rows:
        if mode == 'A':
            _write_od_csv_a(od_rows, coords_csv)
        else:
            _write_od_csv_b(od_rows, raw_coords_csv)
            transformer.transform_csv(raw_coords_csv, coords_csv)

    return coords_csv


def _write_od_csv_a(rows, path):
    """Write OD centroids as panoramic coords (Mode A)."""
    import math
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['point_name', 'image_name', 'cls',
                           'x[px]', 'y[px]', 'angular_w'])
        writer.writeheader()
        for r in rows:
            writer.writerow({'point_name': r['point_name'],
                             'image_name': r['image_name'],
                             'cls':        r.get('cls', ''),
                             'x[px]':      r['u'],
                             'y[px]':      r['v'],
                             'angular_w':  r.get('bbox_norm_w', 0.0) * 2.0 * math.pi})


def _write_od_csv_b(rows, path):
    """Write OD centroids as raw landscape coords (Mode B)."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['point_name', 'image_name', 'cls', 'cam_id',
                           'img_w', 'img_h', 'raw_x[px]', 'raw_y[px]',
                           'angular_w'])
        writer.writeheader()
        for r in rows:
            writer.writerow({'point_name': r['point_name'],
                             'image_name': r['image_name'],
                             'cls':        r.get('cls', ''),
                             'cam_id':     r.get('cam_id', -1),
                             'img_w':      r.get('img_w', 0),
                             'img_h':      r.get('img_h', 0),
                             'raw_x[px]':  r['u'],
                             'raw_y[px]':  r['v'],
                             'angular_w':  r.get('angular_w', 0.0)})
