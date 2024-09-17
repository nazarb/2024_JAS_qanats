# YOLOv9 qanat Object Detection model 

Buławka, Nazarij, Hector A. Orengo, and Iban Berganzo-Besga. 2024. ‘Deep Learning-Based Detection of Qanat Underground Water Distribution Systems Using HEXAGON Spy Satellite Imagery’. Journal of Archaeological Science 171:106053. https://doi.org/10.1016/j.jas.2024.106053.

## Abstract

Qanats are a remarkable type of ancient hydraulic structure for sustainable water distribution in arid environments that use subterranean channels to transport water from highland or mountainous areas. The presence of the qanat system is marked by a line of regularly spaced shafts visible from the surface, which can be used to detect qanats using satellite imagery. Typically, qanats have been documented by field mapping or manual digitisation within a Geographic Information System (GIS) environment. This process is time-consuming due to the numerous shafts within each qanat line. However, several automated methods for detecting qanat structures have been explored, using techniques such as morphological filters, custom convolutional neural networks (CNN) and, more recently, YOLOv5 and Mask R-CNN. These approaches used high-resolution RGB images and CORONA images. However, the use of black and white CORONA in CNNs has been limited in its applicability due to a high rate of false positives.

This paper explores the potential of YOLOv9 in processing the black and white HEXAGON (KH-9) high-resolution spy satellite system launched in 1971. Two areas in Afghanistan (Maiwand) and Iran (Gorgan Plain) were selected to train the system images extracted from HEXAGON imagery and artificial synthetic data. The training dataset was augmented using the Albumentation library, which increased the number of tiles used. The model was tested using two types of HEXAGON imagery for selected areas in Afghanistan (Maiwand), Iran (Gorgan Plain) and Morocco (Rissani), and CORONA imagery in Iran (Gorgan Plain).

Our study provided a model capable of predicting the location of qanat shafts with a precision of over 0.881 and a recall of 0.627 for most of the case studies tested. This is the first case study aimed at detecting qanats in different landscapes using different types of satellite imagery. Using real, augmented, and artificial data allowed us to generalise the representation of qanats into lineal groups of circular features. Thanks to applying labelling for individual qanats and their pairs as separate classes, our approach eliminated most of the isolated and clustered false positives.

# Workflow


## Training

```
!python train_dual.py --device 0 --workers 8 --batch 32 --img 256 --epochs 200 --data c:/ml/yolov9/Datasets/qanats_256_synt_G1_AFG1_pairs_single_2.yaml --weights c:/ml/yolov9/Datasets/yolov9-e.pt --cfg c:/ml/yolov9/models/detect/yolov9_qanat.yaml --hyp c:/ml/yolov9/data/hyps/hyp.scratch-high.yaml
```
## Validation

```
!python val_dual.py --data c:/ml/yolov9/Datasets/qanats_256_synt_G1_AFG1_pairs_single_2.yaml --img 256 --batch 32 --conf 0.001 --iou 0.7 --weights C:/ML/yolov9/runs/train/exp/weights/best.pt --save-json --name yolov9_e_256_val  --device 0
```

## Detection

### Gorgan, Iran

#### D3C1219 HEXAGON images
Raw image:  D3C1219-100139A044_a
AOI: clipped to G2

```
!python detect_dual.py --weights C:/ML/yolov9/runs/train/exp/weights/best.pt --conf 0.1 --source D:/UnderTheSands/Rasters/ML/G2/D3C1219/D3C1219_a3_split/ --img 256 --save-txt --save-conf --max-det 9999999999 --device 0
```
#### D3C1216 HEXAGON images
Raw image:  D3C1216-401091A016_a
AOI: clipped to G2

```
!python detect_dual.py --weights C:/ML/yolov9/runs/train/exp/weights/best.pt --conf 0.1 --source D:/UnderTheSands/Rasters/ML/G2/D3C1216/A016_a3_Split/ --img 256 --save-txt --save-conf --max-det 9999999999 --device 0
```
#### DS1110 CORONA images
Raw image:  ds1110-1089df038_wgs_1_a
AOI: clipped to G3

```
!python detect_dual.py --weights C:/ML/yolov9/runs/train/exp/weights/best.pt --conf 0.1 --source D:/UnderTheSands/Rasters/ML/G3/ds1110-1089df038_wgs_1_c_SPLIT/ --img 256 --save-txt --save-conf --max-det 9999999999 --device 0
```

### Maiwand, Afghanistan

### D3C1209 HEXAGON images
Raw image:  D3C1209-300484A022_a
AOI: clipped to AFG2

```
!python detect_dual.py --weights C:/ML/yolov9/runs/train/exp/weights/best.pt --conf 0.1 --source D:/UnderTheSands/Rasters/ML/AFG2/D3C1209-300484A022_a_10_split/ --img 256 --save-txt --save-conf --max-det 9999999999 --device 0
```

### D3C1209 HEXAGON images
Raw image:  D3C1219-300971A040_c
AOI: clipped to AFG2

```
!python detect_dual.py --weights C:/ML/yolov9/runs/train/exp/weights/best.pt --conf 0.1 --source D:/UnderTheSands/Rasters/ML/AFG2/D3C1219-300971A040_c_10_split/ --img 256 --save-txt --save-conf --max-det 9999999999 --device 0
```
## Rissani, Morocco

### D3C1211 HEXAGON images
Raw image:  D3C1211-100139A008_i, D3C1211-100139A008_j
AOI: clipped to MR1

```
!python detect_dual.py --weights C:/ML/yolov9/runs/train/exp/weights/best.pt --conf 0.1 --source D:/UnderTheSands/Rasters/ML/MR1/D3C1211-100139A008_ij_clip7_b1_SPLIT/ --img 256 --save-txt --save-conf --max-det 9999999999 --device 0
```
### D3C1218 HEXAGON images
Raw image:  D3C1218-401390F004_b, D3C1218-401390F004_c
AOI: clipped to MR2

```
!python detect_dual.py --weights C:/ML/yolov9/runs/train/exp/weights/best.pt --conf 0.1 --source D:/UnderTheSands/Rasters/ML/MR2b/D3C1218-401390F004_bc_8_SPLIT/ --img 256 --save-txt --save-conf --max-det 9999999999 --device 0
```

## Post-processing
* Convert YOLO labels to SHP (developed by [Iban Berganzo-Besga](https://github.com/iberganzo), and adjusted by [Nazarij Buławka](https://github.com/nazarb)
* Convert the results to the same coordinate system (EPSG: 4326)
* Remove duplicates
* Perform spatial join of class 0 and 1
* Perform DBSCAN
* Filter data
* Compare detected features and reference data
* Calculate precision, recall and F1-score
* Export results



# Citation

```

 @article{Buławka_Orengo_Berganzo-Besga_2024, title={Deep learning-based detection of qanat underground water distribution systems using HEXAGON spy satellite imagery}, volume={171}, rights={All rights reserved}, DOI={10.1016/j.jas.2024.106053}, abstractNote={Qanats are a remarkable type of ancient hydraulic structure for sustainable water distribution in arid environ­ ments that use subterranean channels to transport water from highland or mountainous areas. The presence of the qanat system is marked by a line of regularly spaced shafts visible from the surface, which can be used to detect qanats using satellite imagery. Typically, qanats have been documented by field mapping or manual digitisation within a Geographic Information System (GIS) environment. This process is time-consuming due to the numerous shafts within each qanat line. However, several automated methods for detecting qanat structures have been explored, using techniques such as morphological filters, custom convolutional neural networks (CNN) and, more recently, YOLOv5 and Mask R-CNN. These approaches used high-resolution RGB images and CORONA images. However, the use of black and white CORONA in CNNs has been limited in its applicability due to a high rate of false positives.}, journal={Journal of Archaeological Science}, author={Buławka, Nazarij and Orengo, Hector A. and Berganzo-Besga, Iban}, year={2024}, pages={106053}, language={en} }




```
