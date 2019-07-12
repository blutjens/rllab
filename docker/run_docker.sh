#!/bin/bash
{ try
  NV_GPU=0 docker run -it \
  -e DISPLAY=unix$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --name rllab \
  -p 8888:8888 -p 6006:6006 \
  -v /home/$USER/:/home/$USER \
  lutjens/rllab:v0

} || {  catch
  # Remove running instance and try again
  docker rm rllab

  NV_GPU=0 docker run -it \
  -e DISPLAY=unix$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --name rllab \
  -p 8888:8888 -p 6006:6006 \
  -v /home/$USER/:/home/$USER \
  lutjens/rllab:v0
}