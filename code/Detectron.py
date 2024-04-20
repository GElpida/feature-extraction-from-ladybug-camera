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
              self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
              self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
              self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
              self.classes = [9,11] # 9-->traffic light, 10-->fire hydrant, 11-->stop sign, 13-->bench
           elif model_type == 'IS': #instance segmentation
               self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
               self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
               self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
               self.classes = [9,10,11,13]
           elif model_type == 'P': #panoptic
               self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
               self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
               self.stuff_classes = [] #21-->road
               self.thing_classes = [9,10,11,13] # 9-->traffic light, 10-->fire hydrant, 11-->stop sign, 13-->bench
               self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
           else : print('Invalid model type. Valid model type options : OD, IS, P')

        elif self.model == 'Cityscapes':
            if model_type == 'SS': #semantic segmentation
                sys.path.append('../projects')
                from projects.DeepLab.deeplab.config import add_deeplab_config
                add_deeplab_config(self.cfg)
                self.cfg.merge_from_file('../projects/Cityscapes/sem_seg/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml')
                self.cfg.MODEL.WEIGHTS = "./projects/Cityscapes/sem_seg/model_final_a8a355.pkl"
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
                self.cfg.INPUT.CROP.ENABLED = False
            elif model_type == 'P':
                sys.path.append('./projects/detectron2/projects')
                from projects.Panoptic_DeepLab.panoptic_deeplab.config import add_panoptic_deeplab_config
                add_panoptic_deeplab_config(self.cfg)
                self.cfg.merge_from_file('./projects/detectron2/projects/Panoptic_DeepLab/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml')
                self.cfg.MODEL.WEIGHTS = "./projects/Cityscapes/panoptic/model_final_bd324a.pkl"
                self.stuff_classes = [5,6,7] #0-->road, 5-->pole, 6-->traffic light, 7-->traffic sign
                self.thing_classes = [5,6,7] #0-->road, 5-->pole, 6-->traffic light, 7-->traffic sign
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
            else : print('Invalid model type. Valid model type options : SS, P')

        elif self.model == 'Crosswalk':
            if model_type == 'OD':
                self.cfg.merge_from_file('./projects/Crosswalk/output/config.yaml')
                self.cfg.MODEL.WEIGHTS = './projects/Crosswalk/output/model_final.pth'
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
                self.classes = ['','Crosswalk']
            else : print('Invalid model type. Valid model type option : OD')
        
        elif self.model == 'Traffic_Sign':
            if model_type == 'OD':
                self.cfg.merge_from_file('./projects/Traffic_Sign/output/config.yaml')
                self.cfg.MODEL.WEIGHTS = './projects/Traffic_Sign/output/model_final.pth'
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65
                self.classes = ["","Attention","Bend_to_left","Bend_to_right","Crosswalk","Fork_road","Give_way","Narrow_road","No_entry","No_left_turn","No_right_turn","No_u_turn","Roundabout_mandatory","Speed_limit_100KM","Speed_limit_110KM","Speed_limit_120KM","Speed_limit_20KM","Speed_limit_30KM","Speed_limit_40KM","Speed_limit_50KM","Speed_limit_60KM","Speed_limit_70KM","Speed_limit_80KM","Speed_limit_90KM","Stop"]
            else : print('Invalid model type. Valid model type option : OD')

        elif self.model == 'Safety_Cones':
            if model_type == 'OD':
                self.cfg.merge_from_file('./projects/Safety_Cones/output/config.yaml')
                self.cfg.MODEL.WEIGHTS = './projects/Safety_Cones/output/model_final.pth'
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
    
    def output(self, im, outputs, name, directory):
    
        im_shape = im.shape

        new_im_shape = max(im_shape[0],im_shape[1])
        
        if self.model_type == 'OD':
            
            info = {}

            #Export information from image name and store them in a list
            split_image_name = name.split('_')

            stream_name = split_image_name[0]+'_'+split_image_name[1]+'_'+split_image_name[2]
            image_id = split_image_name[3]
            Cam_id = split_image_name[4]

            for i in [0,1,2,3,4,5]:
                if Cam_id == 'Cam'+str(i):
                    Cam_id = i

            info.update({'stream_name':stream_name, 'image_id':image_id, 'Cam_id':Cam_id, 'instances': outputs})
            
            #Save dictionary as a csv file
            csv_name = directory +'/'+name+'_'+self.model+'_OD.csv' #specify csv name

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
            
            for info in my_segments_info:
                id = info['id']
                
                #Remove disabled classes
                mask = torch.zeros(panoptic_seg.shape)
                mask[torch.where(panoptic_seg==id)] = torch.ones(len(torch.where(panoptic_seg==id)[0]))

                #Νormalizes mask in range 0 - 255
                mask = cv2.normalize(cv2.UMat(np.float32(mask)), None, 0, 255, cv2.NORM_MINMAX)
                
                #Specify mask's name
                model = self.model 
                if 'instance_id' not in info.keys():
                    info.update({'instance_id':'0'})
                out_name = name+'_'+model+'_'+str(info['category_id'])+'_'+str(info['instance_id'])+'_mask.jpg' 
                cv2.imwrite(os.path.join(directory, out_name), mask) #save the output mask

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

