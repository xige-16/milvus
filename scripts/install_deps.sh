#!/usr/bin/env bash

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function install_linux_deps() {
  if [[ -x "$(command -v apt)" ]]; then
      # for Ubuntu 18.04
      sudo apt install -y g++ gcc make lcov libtool m4 autoconf automake ccache libssl-dev zlib1g-dev libboost-regex-dev \
          libboost-program-options-dev libboost-system-dev libboost-filesystem-dev \
          libboost-serialization-dev python3-dev libboost-python-dev libcurl4-openssl-dev gfortran libtbb-dev libzstd-dev libaio-dev \
          uuid-dev libpulse-dev
  elif [[ -x "$(command -v yum)" ]]; then
      # for CentOS 8
      sudo yum install -y epel-release && \
      sudo yum install -y git make lcov libtool m4 autoconf automake ccache openssl-devel zlib-devel libzstd-devel \
          libcurl-devel python3-devel\
          devtoolset-8-gcc devtoolset-8-gcc-c++ devtoolset-8-gcc-gfortran \
          clang clang-tools-extra libuuid-devel pulseaudio-libs-devel

      echo "source scl_source enable devtoolset-8" | sudo scl enable devtoolset-8 -- bash

      # Install tbb
      git clone https://github.com/wjakob/tbb.git && \
      cd tbb/build && \
      cmake .. && make -j && \
      sudo make install && \
      cd ../../ && rm -rf tbb/

      # Install boost
      wget -q https://boostorg.jfrog.io/artifactory/main/release/1.65.1/source/boost_1_65_1.tar.gz && \
          tar zxf boost_1_65_1.tar.gz && cd boost_1_65_1 && \
          ./bootstrap.sh --prefix=/usr/local --with-toolset=gcc --without-libraries=python && \
          sudo ./b2 -j2 --prefix=/usr/local --without-python toolset=gcc install && \
          cd ../ && rm -rf ./boost_1_65_1*
  else
      echo "Error Install Dependencies ..."
      exit 1
  fi
}

function install_mac_deps() {
  sudo xcode-select --install > /dev/null 2>&1
  brew install libomp ninja cmake llvm@15 ccache grep
  export PATH="/usr/local/opt/grep/libexec/gnubin:$PATH"
  brew update && brew upgrade && brew cleanup

  if [[ $(arch) == 'arm64' ]]; then
    brew install openssl
    brew install librdkafka
    brew install pkg-config
  fi
}

if ! command -v go &> /dev/null
then
    echo "go could not be found, please install it"
    exit
fi

if ! command -v cmake &> /dev/null
then
    echo "cmake could not be found, please install it"
    exit
fi

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     install_linux_deps;;
    Darwin*)    install_mac_deps;;
    *)          echo "Unsupported OS:${unameOut}" ; exit 0;
esac

