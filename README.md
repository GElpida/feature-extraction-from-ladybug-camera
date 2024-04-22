# feature-extraction-from-ladybug-camera
For this project a methodology was developed that utilizes data from a Mobile Mapping System -which uses a multi-camera Ladybug 5+ as a mapping sensor- to extract features from the Ladybug images and determine their spatial coordinates.

The methodolody uses the [Detectron 2](https://github.com/facebookresearch/detectron2) Computer Vision Framework.

Image 1 | Image 2
---|--- 
![20344317_20220413_095709_000000_Cam1_output](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/4ce13c9e-8763-4742-9194-af0975926f1d) | ![20344317_20220413_095709_000002_Cam0_output](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/3e5a2ee8-ee99-492f-8041-6f342660471e)

## Ladybug 5+
![ladybug5plus-frt-red](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/83de8cfa-3f98-4303-8c37-20e5e7db9a97)

The multi-camera [Ladybug 5+](https://www.flir.com/products/ladybug5plus/?vertical=machine+vision&segment=iis) is a product of Teledyne FLIR suitable for mobile mapping.

## Installation 
1. [Install Detectron 2](https://haroonshakeel.medium.com/detectron2-setup-on-windows-10-and-linux-407e5382df1)
2. Clone this repository
3. Create folder [projects](projects.md)
