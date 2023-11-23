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

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
ROOT_DIR="$( cd -P "$( dirname "$SOURCE" )/.." && pwd )"
CPP_SRC_DIR="${ROOT_DIR}/internal/core"
BUILD_OUTPUT_DIR="${ROOT_DIR}/cmake_build"

if [[ ! -d ${BUILD_OUTPUT_DIR} ]]; then
  mkdir ${BUILD_OUTPUT_DIR}
fi

source ${ROOT_DIR}/scripts/setenv.sh
pushd ${BUILD_OUTPUT_DIR}

export CONAN_REVISIONS_ENABLED=1
export CXXFLAGS="-Wno-error=address -Wno-error=deprecated-declarations"
export CFLAGS="-Wno-error=address -Wno-error=deprecated-declarations"
if [[ ! `conan remote list` == *default-conan-local* ]]; then
    conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local
fi
unameOut="$(uname -s)"
case "${unameOut}" in
  Darwin*)
    conan install ${CPP_SRC_DIR} --install-folder conan --build=missing -s compiler=clang -s compiler.version=${llvm_version} -s compiler.libcxx=libc++ -s compiler.cppstd=17 || { echo 'conan install failed'; exit 1; }
    ;;
  Linux*)
    echo "Running on ${OS_NAME}"
    export CPU_TARGET=avx
    GCC_VERSION=`gcc -dumpversion`
    if [[ `gcc -v 2>&1 | sed -n 's/.*\(--with-default-libstdcxx-abi\)=\(\w*\).*/\2/p'` == "gcc4" ]]; then
      conan install ${CPP_SRC_DIR} --install-folder conan --build=missing -s compiler.version=${GCC_VERSION} || { echo 'conan install failed'; exit 1; }
    else
      conan install ${CPP_SRC_DIR} --install-folder conan --build=missing -s compiler.version=${GCC_VERSION} -s compiler.libcxx=libstdc++11 || { echo 'conan install failed'; exit 1; }
    fi
    ;;
  *)
    echo "Cannot build on windows"
    ;;
esac

popd

pushd ${ROOT_DIR}/cmake_build/thirdparty

git clone --depth=1 --branch v0.42.0 https://github.com/apache/incubator-opendal.git opendal
cd opendal
if command -v cargo >/dev/null 2>&1; then
    echo "cargo exists"
else
    bash -c "curl https://sh.rustup.rs -sSf | sh -s -- -y" || { echo 'rustup install failed'; exit 1;}
    source $HOME/.cargo/env
fi
pushd bindings/c
cargo build --release --verbose || { echo 'opendal_c build failed'; exit 1; }
popd
mkdir -p ${ROOT_DIR}/internal/core/output/lib
mkdir -p ${ROOT_DIR}/internal/core/output/include
cp target/release/libopendal_c.a ${ROOT_DIR}/internal/core/output/lib/libopendal_c.a
cp bindings/c/include/opendal.h ${ROOT_DIR}/internal/core/output/include/opendal.h

popd
