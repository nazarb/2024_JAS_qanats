#
```
docker build -t yolov9sam .
```
#

```
docker run --gpus all -it --name yolov9 -v /media/work:/media -p 8877:8877 -p 9433:9433 --shm-size=32g yolov9sam
```
