import warnings
warnings.filterwarnings("ignore") 

# import some common libraries
import sys, os, distutils.core
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
              self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection\\faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
              self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection\\faster_rcnn_X_101_32x8d_FPN_3x.yaml")
              self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
              self.classes = [2,9,11] # 0-->person, 2-->car, 9-->traffic light, 10-->fire hydrant, 11-->stop sign, 13-->bench
           elif model_type == 'IS': #instance segmentation
               self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation\\mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
               self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation\\mask_rcnn_X_101_32x8d_FPN_3x.yaml")
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
               sys.path.append('..\\projects')
               from projects.Mask2Former.mask2former.config import add_maskformer2_config # Import MaskFormer configs
               add_maskformer2_config(self.cfg)
               self.cfg.merge_from_file('..\\projects\\MaskFormer\\panoptic\\maskformer2_R50_bs16_50ep.yaml')
               self.cfg.MODEL.WEIGHTS = "..\\projects\\MaskFormer\\panoptic\\model_final_94dc52.pkl"
               self.stuff_classes = [40] #21-->road, 28-->house, 29-->light, 37-->tree, 40-->sky, 44-->pavement, 45-->mountain, 50-->building, 52-->wall
               self.thing_classes = [0,2] # 0-->person, 1--> bicycle, 2-->car, 3-->motorcycle, 5-->bus, 6-->train, 7-->truck, 9-->traffic light, 10-->fire hydrant, 11-->stop sign, 13-->bench
               self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 1
            else : print('Invalid model type. Valid model type options : P')

        elif self.model == 'Cityscapes':
            if model_type == 'SS': #semantic segmentation
                sys.path.append('..\\projects')
                from projects.DeepLab.deeplab.config import add_deeplab_config
                add_deeplab_config(self.cfg)
                self.cfg.merge_from_file('..\\projects\\Cityscapes\\sem_seg\\deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml')
                self.cfg.MODEL.WEIGHTS = "..\\projects\\Cityscapes\\sem_seg\\model_final_a8a355.pkl"
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
                self.cfg.INPUT.CROP.ENABLED = False
            elif model_type == 'P':
                sys.path.append('..\\projects')
                from projects.Panoptic_DeepLab.panoptic_deeplab.config import add_panoptic_deeplab_config
                add_panoptic_deeplab_config(self.cfg)
                self.cfg.merge_from_file('..\\projects\\Cityscapes\\panoptic\\panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml')
                self.cfg.MODEL.WEIGHTS = "..\\projects\\Cityscapes\\panoptic\\model_final_bd324a.pkl"
                self.stuff_classes = [5] #0-->road, 5-->pole, 6-->traffic light, 7-->traffic sign
                self.thing_classes = [5] #0-->road, 5-->pole, 6-->traffic light, 7-->traffic sign
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
            else : print('Invalid model type. Valid model type options : SS, P')

        elif self.model == 'Crosswalk':
            if model_type == 'OD':
                self.cfg.merge_from_file('..\\projects\\Crosswalk\\output\\config.yaml')
                self.cfg.MODEL.WEIGHTS = '..\\projects\\Crosswalk\\output\\model_final.pth'
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
                self.classes = ['','Crosswalk']
            else : print('Invalid model type. Valid model type option : OD')
        
        elif self.model == 'Traffic_Sign':
            if model_type == 'OD':
                self.cfg.merge_from_file('..\\projects\\Traffic_Sign\\output\\config.yaml')
                self.cfg.MODEL.WEIGHTS = '..\\projects\\Traffic_Sign\\output\\model_final.pth'
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65
                self.classes = ["","Attention","Bend_to_left","Bend_to_right","Crosswalk","Fork_road","Give_way","Narrow_road","No_entry","No_left_turn","No_right_turn","No_u_turn","Roundabout_mandatory","Speed_limit_100KM","Speed_limit_110KM","Speed_limit_120KM","Speed_limit_20KM","Speed_limit_30KM","Speed_limit_40KM","Speed_limit_50KM","Speed_limit_60KM","Speed_limit_70KM","Speed_limit_80KM","Speed_limit_90KM","Stop"]
            else : print('Invalid model type. Valid model type option : OD')

        elif self.model == 'Safety_Cones':
            if model_type == 'OD':
                self.cfg.merge_from_file('..\\projects\\Safety_Cones\\output\\config.yaml')
                self.cfg.MODEL.WEIGHTS = '..\\projects\\Safety_Cones\\output\\model_final.pth'
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
                self.classes = ['','Safety_Cone']
            else : print('Invalid model type. Valid model type option : OD')
        
        else :  print('Invalid model. Valid model options : COCO, Cityscapes, Crosswalk, Traffic_Sign, Safety_Cones') 

        self.cfg.MODEL.DEVICE = 'cpu' # cpu or cuda
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
    
        im_shape = im.shape

        new_im_shape = max(im_shape[0],im_shape[1])
        
        if self.model_type == 'OD':
            
            info = {}

            #Export information from image name and store them in a list
        #    split_image_name = name.split('_')

        #    image_id = split_image_name[3]
        #    Cam_id = split_image_name[4]

        #    for i in [0,1,2,3,4,5]:
        #        if Cam_id == 'Cam'+str(i):
        #            Cam_id = i

        #    info.update({'stream_name':stream_name, 'image_id':image_id, 'Cam_id':Cam_id, 'instances': outputs})
            info.update({'instances': outputs})
            
            #Save dictionary as a csv file
            csv_name = directory +'\\'+name+'_'+self.model+'_OD.csv' #specify csv name

            with open(csv_name, 'w') as csvfile:
                keys = info.keys()
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                writer.writerow(info)

        if self.model_type == 'P':

            for row in outputs:
                panoptic_seg = row['panoptic_seg']
                my_segments_info = row['my_segments_info']

            #Remove padding from mask 
            b1 = 0 + int((new_im_shape - im_shape[0])/2) 
            b3 = 0 + int((new_im_shape - im_shape[1])/2)
    
            if panoptic_seg.shape != im_shape:
               panoptic_seg = panoptic_seg[b1:b1+im_shape[0], b3:b3+im_shape[1]]
            

            if mode == "mask":
                for info in my_segments_info:
                    seg_id = info['id']
                    mask = (panoptic_seg == seg_id).to(torch.uint8)
                    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                    model = self.model
                    if 'instance_id' not in info:
                        info['instance_id'] = '0'
                    out_name = f"{name}_{model}_{info['category_id']}_{info['instance_id']}.jpg"
                    out_path = os.path.join(directory, out_name)
                    cv2.imwrite(out_path, mask_np)

            elif mode == "blur":
                # Combine all panoptic masks into one
                combined_mask = torch.zeros_like(panoptic_seg, dtype=torch.uint8)
                for info in my_segments_info:
                    seg_id = info['id']
                    combined_mask[panoptic_seg == seg_id] = 1

                # Optional external mask
                external_mask_path = ""  # specify path to external mask if needed

                ext_mask = None
                if external_mask_path:
                    ext_mask = cv2.imread(external_mask_path, cv2.IMREAD_GRAYSCALE)

                    if ext_mask is not None:
                        # Resize if needed
                        if ext_mask.shape != combined_mask.shape:
                            ext_mask = cv2.resize(
                                ext_mask,
                                (combined_mask.shape[1], combined_mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )

                        # Convert to binary
                        ext_mask_bin = (ext_mask > 0).astype(np.uint8)
                        ext_mask_torch = torch.from_numpy(ext_mask_bin)

                        # Intersect masks
                        combined_mask = torch.min(combined_mask, ext_mask_torch)
                    else:
                        print("Warning: External mask could not be loaded. Ignoring it.")

                # Apply blur
                im_blur = im.copy()
                blurred = cv2.GaussianBlur(im, (31, 31), 0)
                mask_np = (combined_mask.cpu().numpy() * 255).astype(np.uint8)
                im_blur[mask_np == 255] = blurred[mask_np == 255]

                # Apply black outside external mask ONLY if it exists
                if ext_mask is not None:
                    im_blur[ext_mask == 0] = 0

                # Save final image
                out_name = f"{name}_{self.model}_blur.jpg"
                out_path = os.path.join(directory, out_name)
                cv2.imwrite(out_path, im_blur)

            v = Visualizer(im[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
            out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), my_segments_info, area_threshold=None, alpha=0.7)
        
        elif self.model_type == 'SS':
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
            out = v.draw_sem_seg(outputs.to("cpu"), area_threshold=None, alpha=0.8)
        
        elif self.model in ['Crosswalk', 'Traffic_Sign', 'Safety_Cones']:
             MetadataCatalog.get("my_dataset_train").thing_classes = self.classes
             metadata = MetadataCatalog.get("my_dataset_train")
             v = Visualizer(im[:,:,::-1], metadata, scale=1)
             out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
             MetadataCatalog.remove("my_dataset_train")
        
        else :
            v = Visualizer(im[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
            for i in self.classes:
                out = v.draw_instance_predictions(outputs["instances"][outputs["instances"].pred_classes == i].to("cpu"))
 
        output = out.get_image()[:,:,::-1] 

        return output

