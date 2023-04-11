# Detection using SSD (2016)

#### Requisites
- images_framework https://github.com/bobetocalo/images_framework

#### Installation
This repository must be located inside the following directory:
```
images_framework
    └── detection
        └── ssd16_detection
```
#### Usage
```
usage: ssd16_detection_test.py [-h] [--input-data INPUT_DATA] [--show-viewer] [--save-image]
```

* Use the --input-data option to set an image, directory, camera or video file as input.

* Use the --show-viewer option to show results visually.

* Use the --save-image option to save the processed images.
```
usage: Detection --database DATABASE [--shapefile]
```

* Use the --database option to select the database model.

* Use the --shapefile option to save results as vector data.
```
usage: SSD16Detection [--gpu GPU]
```

* Use the --gpu option to set the GPU identifier (negative value indicates CPU mode).
```
> python images_framework/detection/ssd16_detection/test/ssd16_detection_test.py --input-data images_framework/detection/ssd16_detection/test/example.tif --database AFLW --gpu 0 --save-image
```
