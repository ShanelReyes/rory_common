

# Install Pyfhel

 python3 setup.py build_clib


 # 1. Clone and build SEAL
git clone https://github.com/microsoft/SEAL.git
cd SEAL
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=OFF    \
      -DSEAL_USE_ZSTD=ON         \
      -DSEAL_USE_MSGSL=ON
cmake --build build -j$(nproc)

# 2. Install headers + static lib to /usr/local
sudo cmake --install build
