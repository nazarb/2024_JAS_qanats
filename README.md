# YOLOv9 qanat Object Detection model 

Buławka, Nazarij, Hector A. Orengo, and Iban Berganzo-Besga. 2024. ‘Deep Learning-Based Detection of Qanat Underground Water Distribution Systems Using HEXAGON Spy Satellite Imagery’. Journal of Archaeological Science JASC24-174.

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
## Citation

```

@article{bulawkaDeepLearningbasedDetection2024,
	title = {Deep {Learning}-based detection of {Qanat} underground water distribution systems using {HEXAGON} spy satellite imagery},
	volume = {JASC24-174},
	journal = {Journal of Archaeological Science},
	author = {Buławka, Nazarij and Orengo, Hector A. and Berganzo-Besga, Iban},
	year = {2024},
	note = {paper international sent to publication},
}



```
