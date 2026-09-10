#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --ssh default=$HOME/.ssh/id_rsa -t ssd16_detection_image .
sudo docker run --name ssd16_detection_container --rm --gpus all -it -d ssd16_detection_image bash
sudo docker exec -w /home/username/ssd16_detection ssd16_detection_container python test/ssd16_detection_test.py --input-data test/example.tif --database aflw --gpu 0 --save-image
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo docker cp ssd16_detection_container:/home/username/conda/envs/ssd16/lib/python3.8/site-packages/images_framework/output/images/. output/
sudo chown -R "${USER}":"${USER}" output
sudo docker rm -f ssd16_detection_container
sudo docker image rm ssd16_detection_image
sudo docker builder prune -a -f