SCRIPT_DIR=$(cd $(dirname $0); pwd)
xhost +

docker run --rm -it \
--gpus all \
-u `id -u`:`id -g` \
-v /etc/passwd:/etc/passwd:ro \
-v /etc/group:/etc/group:ro \
-v $SCRIPT_DIR/homedir:/home/`whoami` \
-v `pwd`:/workdir \
-v $HOME/.Xauthority:$HOME/.Xauthority \
--env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--privileged \
-w /workdir groundingdino bash

# --env="TORCH_CUDA_ARCH_LIST=\"6.1\"" \