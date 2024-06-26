#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t ssd16_detection_image .
sudo docker volume create --name ssd16_detection_volume
sudo docker run --name ssd16_detection_container -v ssd16_detection_volume:/home/username --rm --gpus all -it -d ssd16_detection_image bash
sudo docker exec -w /home/username/ ssd16_detection_container python images_framework/detection/ssd16_detection/test/ssd16_detection_test.py --input-data images_framework/detection/ssd16_detection/test/example.tif --database aflw --gpu 0 --save-image
sudo docker stop ssd16_detection_container
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo chown -R "${USER}":"${USER}" /var/lib/docker/
rsync --delete -azvv /var/lib/docker/volumes/ssd16_detection_volume/_data/images_framework/output/images/ output
sudo docker system prune --all --force --volumes
sudo docker volume rm $(sudo docker volume ls -qf dangling=true)
