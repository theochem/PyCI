set -xe


# # NIGHTLY_FLAG=""

# # if [ "$#" -eq 1 ]; then
# #     PROJECT_DIR="$1"
# # elif [ "$#" -eq 2 ] && [ "$1" = "--nightly" ]; then
# #     NIGHTLY_FLAG="--nightly"
# #     PROJECT_DIR="$2"
# # else
# #     echo "Usage: $0 [--nightly] <project_dir>"
# #     exit 1
# # fi

# printenv
# # Update license
# cat $PROJECT_DIR/tools/wheels/LICENSE_linux.txt >> $PROJECT_DIR/LICENSE.txt

# TODO: delete along with enabling build isolation by unsetting
# CIBW_BUILD_FRONTEND when scipy is buildable under free-threaded
# python with a released version of cython
# FREE_THREADED_BUILD="$(python -c"import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))")"
# if [[ $FREE_THREADED_BUILD == "True" ]]; then
#     python -m pip install -U --pre pip
#     python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple numpy cython
#     python -m pip install git+https://github.com/serge-sans-paille/pythran
#     python -m pip install ninja meson-python pybind11
# fi

yum install -y git make

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