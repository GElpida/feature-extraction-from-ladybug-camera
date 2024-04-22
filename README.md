# feature-extraction-from-ladybug-camera
For this project a methodology was developed that utilizes data from a Mobile Mapping System -which uses a multi-camera Ladybug 5+ as a mapping sensor- to extract features from the Ladybug images and determine their spatial coordinates.

The methodolody uses the [Detectron 2](https://github.com/facebookresearch/detectron2) Computer Vision Framework.

Output 1 | Output 2
---|--- 
![20344317_20220413_095709_000005_Cam0_output](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/8d0d418e-ae61-4b1e-bdcf-d14cfb379736) | ![20344317_20220413_095709_000007_Cam1_output](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/c644c7d6-a202-49ce-b058-ef8507033075)

Centroid extraction process |
:---: |
![image](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/4237f4b4-6ae3-4bc9-8961-5c7f0627c357) |

## Ladybug 5+
![ladybug5plus-frt-red](https://github.com/GElpida/feature-extraction-from-ladybug-camera/assets/162966788/83de8cfa-3f98-4303-8c37-20e5e7db9a97)

The multi-camera [Ladybug 5+](https://www.flir.com/products/ladybug5plus/?vertical=machine+vision&segment=iis) is a product of Teledyne FLIR suitable for mobile mapping.

## Installation 
1. [Install Detectron 2](https://haroonshakeel.medium.com/detectron2-setup-on-windows-10-and-linux-407e5382df1)
2. Clone this repository
3. Create folder [projects](projects.md)
