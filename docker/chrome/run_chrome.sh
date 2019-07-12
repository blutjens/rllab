docker rm lutjens/chrome

docker run -it \
    --net host \
    --cpuset-cpus 0 \
    --memory 512mb \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=unix$DISPLAY \
    -v $HOME/Downloads:/root/Downloads \
    -v $HOME/.config/google-chrome/:/data \
    --device /dev/snd \
    --name chrome \
    lutjens/chrome

sudo docker run -it \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=unix$DISPLAY \
--name cathode \
lutjens/cathode 

xhost +local:docker 