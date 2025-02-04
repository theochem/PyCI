set -xe

if ldd --version 2>&1 | grep -q "musl"; then
    echo "Inside a musllinux container"
    apk add git make
else
    echo "Inside a manylinux container"
    yum install -y git make
fi



# download dependencies
# echo $PROJECT_DIR/deps
# mkdir -p $PROJECT_DIR/deps
# cd $PROJECT_DIR/deps
# git clone https://gitlab.com/libeigen/eigen.git
# git clone https://gitlab.com/libeigen/eigen.git
# git clone https://github.com/greg7mdp/parallel-hashmap.git
# git clone https://github.com/pybind/pybind11.git
# cd $PROJECT_DIR

# install python dependencies
python -m pip install -U --pre pip
#TODO: doubt, what will happen if the numpy version is different from the one in the wheel
python -m pip install numpy
python -m pip install scipy
python -m pip install pytest


# compile library
# make
# make test