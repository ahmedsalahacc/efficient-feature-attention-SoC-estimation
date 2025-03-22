# Get PyTorch paths and flags
TORCH_PATH=~/libtorch
TORCH_INCLUDE="${TORCH_PATH}/include"
TORCH_LIB="${TORCH_PATH}/lib"

# Compile with g++
g++ -c main.cpp \
    -I${TORCH_INCLUDE} \
    -I${TORCH_INCLUDE}/torch/csrc/api/include \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -std=c++17

# Link
g++ main.o -o myprogram \
    -L${TORCH_LIB} \
    -ltorch \
    -ltorch_cpu \
    -lc10

# Set library path and run
export LD_LIBRARY_PATH=${TORCH_LIB}:$LD_LIBRARY_PATH
./myprogram