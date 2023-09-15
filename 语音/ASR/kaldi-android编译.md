# kaldi android编译

**从 NDK r19 开始，NDK 默认安装的工具链可供使用。与任意构建系统进行交互时不再需要使用 `make_standalone_toolchain.py` 脚本**

## ndk r17 32位编译

### 1. 安装android编译工具链

下载[ndk](https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip)并解压

```shell
# 安装64位工具链
$NDK/build/tools/make_standalone_toolchain.py --arch arm64 --api 21  --install-dir=arm64-toolchain
# 安装32位工具链
$NDK/build/tools/make_standalone_toolchain.py --arch arm --api 21 --stl=libc++ --install-dir /tmp/my-android-toolchain
```

https://developer.android.com/ndk/guides/other_build_systems

### 2. 编译android版本的openblas

```shell
git clone https://github.com/xianyi/OpenBLAS
sudo apt-get install gfortran
export ANDROID_TOOLCHAIN_PATH=/tmp/my-android-toolchain
export PATH=${ANDROID_TOOLCHAIN_PATH}/bin:$PATH
#编译32位
export CLANG_FLAGS="-target arm-linux-androideabi -marm -mfpu=vfp -mfloat-abi=softfp --sysroot ${ANDROID_TOOLCHAIN_PATH}/sysroot -gcc-toolchain ${ANDROID_TOOLCHAIN_PATH}"
make TARGET=ARMV7 ONLY_CBLAS=1 AR=ar CC="clang ${CLANG_FLAGS}" HOSTCC=gcc ARM_SOFTFP_ABI=1 USE_THREAD=0 NUM_THREADS=32 -j4
make install NO_SHARED=1 PREFIX=`pwd`/install
#编译64位
export CLANG_FLAGS="-target aarch64-linux-android --sysroot ${ANDROID_TOOLCHAIN_PATH}/sysroot -gcc-toolchain ${ANDROID_TOOLCHAIN_PATH}"
make TARGET=ARMV8 ONLY_CBLAS=1 AR=ar CC="clang ${CLANG_FLAGS}" HOSTCC=gcc -j4
make install NO_SHARED=1 PREFIX=`pwd`/install
```

### 3. 编译CLAPACK

```shell
git clone https://github.com/simonlynen/android_libs.git

cd android_libs/lapack

# remove some compile instructions related to tests
sed -i 's/LOCAL_MODULE:= testlapack/#LOCAL_MODULE:= testlapack/g' jni/Android.mk
sed -i 's/LOCAL_SRC_FILES:= testclapack.cpp/#LOCAL_SRC_FILES:= testclapack.cpp/g' jni/Android.mk
sed -i 's/LOCAL_STATIC_LIBRARIES := lapack/#LOCAL_STATIC_LIBRARIES := lapack/g' jni/Android.mk
sed -i 's/include $(BUILD_SHARED_LIBRARY)/#include $(BUILD_SHARED_LIBRARY)/g' jni/Android.mk
sed -i 's/minSdkVersion="10"/minSdkVersion="21"/g' AndroidManifest.xml
sed -i 's/gnustl_static/c++_static/g' jni/Application.mk
sed -i 's/armeabi armeabi-v7a/armeabi-v7a/g' jni/Application.mk
sed -i 's/android-10/android-21/g' project.properties

# build for android
/path/to/android-ndk-r17c/ndk-build
```

**拷贝`obj/local/armeabi-v7a/` 中的库到OpenBLAS 安装库目录 (e.g: OpenBlas/install/lib)**. kaldi会在该目录寻找 libf2c.a, liblapack.a, libclapack.a,libblas.a.



### 4. 编译openfst

```shell
git clone https://github.com/xianyi/OpenBLAS
sudo apt-get install gfortran
export ANDROID_TOOLCHAIN_PATH=/tmp/my-android-toolchain
export PATH=${ANDROID_TOOLCHAIN_PATH}/bin:$PATH
wget -T 10 -t 1 http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.7.tar.gz 
tar -zxvf openfst-1.6.7.tar.gz
cd openfst-1.6.7/
CXX=clang++ ./configure --prefix=`pwd` --enable-static --enable-shared --enable-far --enable-ngram-fsts --host=armv7a-linux-androideabi LIBS="-ldl"
make -j 4
make install
cd ..
ln -s openfst-1.6.7 openfst
```

### 5. 编译kaldi

```shell
cd kaldi/tools
# 链接openfst安装目录
ln -s ../../openfst-1.6.7 openfst

export PATH=/tmp/my-android-toolchain/bin:$PATH

cd kaldi/src
#You may want to compile Kaldi without debugging symbols.
#In this case, do:
sed -i 's/-g # -O0 -DKALDI_PARANOID/-O3 -DNDEBUG/g' kaldi.mk
# 删除test，在编译android的时候会运行test，在linux上会运行失败
rm -rf `find -name "*test.cc"`
sed -i 's/TESTFILES/#TESTFILES/g' matrix/Makefile

CXX=clang++ ./configure --static --android-incdir=/tmp/my-android-toolchain/sysroot/usr/include/ --host=arm-linux-androideabi --openblas-root=/path/to/OpenBLAS/install --use-cuda=no
make depend -j4
make -j4
```

## ndk r17 64位编译

### 1. 安装android编译工具链

下载[ndk](https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip)并解压

```shell
# 安装64位工具链
$NDK/build/tools/make_standalone_toolchain.py --arch arm64 --api 21  --install-dir=/tmp/my-android-toolchain64
```

https://developer.android.com/ndk/guides/other_build_systems

### 2. 编译android版本的openblas

```shell
git clone https://github.com/xianyi/OpenBLAS
sudo apt-get install gfortran
export ANDROID_TOOLCHAIN_PATH=/tmp/my-android-toolchain64
export PATH=${ANDROID_TOOLCHAIN_PATH}/bin:$PATH
#编译64位
export CLANG_FLAGS="-target aarch64-linux-android --sysroot ${ANDROID_TOOLCHAIN_PATH}/sysroot -gcc-toolchain ${ANDROID_TOOLCHAIN_PATH}"
make TARGET=ARMV8 ONLY_CBLAS=1 AR=ar CC="clang ${CLANG_FLAGS}" HOSTCC=gcc -j4
make install NO_SHARED=1 PREFIX=`pwd`/install64
```

### 3. 编译CLAPACK

```shell
git clone https://github.com/simonlynen/android_libs.git

cd android_libs/lapack

# remove some compile instructions related to tests
sed -i 's/LOCAL_MODULE:= testlapack/#LOCAL_MODULE:= testlapack/g' jni/Android.mk
sed -i 's/LOCAL_SRC_FILES:= testclapack.cpp/#LOCAL_SRC_FILES:= testclapack.cpp/g' jni/Android.mk
sed -i 's/LOCAL_STATIC_LIBRARIES := lapack/#LOCAL_STATIC_LIBRARIES := lapack/g' jni/Android.mk
sed -i 's/include $(BUILD_SHARED_LIBRARY)/#include $(BUILD_SHARED_LIBRARY)/g' jni/Android.mk
sed -i 's/minSdkVersion="10"/minSdkVersion="21"/g' AndroidManifest.xml
sed -i 's/gnustl_static/c++_static/g' jni/Application.mk
sed -i 's/armeabi armeabi-v7a/arm64-v8a/g' jni/Application.mk
sed -i 's/android-10/android-21/g' project.properties

# build for android
/path/to/android-ndk-r17c/ndk-build
```

**拷贝`obj/local/arme64-v8a/` 中的库到OpenBLAS 安装库目录 (e.g: OpenBlas/install/lib)**. kaldi会在该目录寻找 libf2c.a, liblapack.a, libclapack.a,libblas.a.

### 4. 编译openfst

```shell
export ANDROID_TOOLCHAIN_PATH=/tmp/my-android-toolchain
export PATH=${ANDROID_TOOLCHAIN_PATH}/bin:$PATH
wget -T 10 -t 1 http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.7.tar.gz 
tar -zxvf openfst-1.6.7.tar.gz
cd openfst-1.6.7/
CXX=clang++ ./configure --prefix=`pwd` --enable-static --enable-shared --enable-ngram-fsts --host=aarch64-linux-android LIBS="-ldl"
make -j 4
make install
cd ..
ln -s openfst-1.6.7 openfst
```

### 5. 编译kaldi

```shell
cd kaldi/tools
# 链接openfst安装目录
ln -s ../../openfst-1.6.7 openfst

export PATH=/tmp/my-android-toolchain/bin:$PATH

cd kaldi/src
#You may want to compile Kaldi without debugging symbols.
#In this case, do:
sed -i 's/-g # -O0 -DKALDI_PARANOID/-O3 -DNDEBUG/g' kaldi.mk
# 删除test，在编译android的时候会运行test，在linux上会运行失败
rm -rf `find -name "*test.cc"`
sed -i 's/TESTFILES/#TESTFILES/g' matrix/Makefile

CXX=clang++ ./configure --static --android-incdir=/tmp/my-android-toolchain64/sysroot/usr/include/ --host=aarch64-linux-android --openblas-root=/path/to/OpenBLAS/install --use-cuda=no
make depend -j4
make -j4
```

## ndk r22 32位编译

### 1. 编译android版本的openblas

```shell
git clone https://github.com/xianyi/OpenBLAS
sudo apt-get install gfortran

export ANDROID_NDK=/path/to/android-ndk-r22b
export TOOLCHAIN=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64

# 32位
export API=21
export TARGET=armv7a-linux-androideabi
make TARGET=ARMV7 ONLY_CBLAS=1 AR=$TOOLCHAIN/bin/llvm-ar RANLIB=$TOOLCHAIN/bin/llvm-ranlib CC=$TOOLCHAIN/bin/$TARGET$API-clang HOSTCC=gcc ARM_SOFTFP_ABI=1 USE_THREAD=0 NUM_THREADS=32 -j4
make install TARGET=ARMV7 NO_SHARED=1 PREFIX=`pwd`/install
```

### 2. 编译CLAPACK

```shell
git clone https://github.com/simonlynen/android_libs.git

cd android_libs/lapack

# remove some compile instructions related to tests
sed -i 's/LOCAL_MODULE:= testlapack/#LOCAL_MODULE:= testlapack/g' jni/Android.mk
sed -i 's/LOCAL_SRC_FILES:= testclapack.cpp/#LOCAL_SRC_FILES:= testclapack.cpp/g' jni/Android.mk
sed -i 's/LOCAL_STATIC_LIBRARIES := lapack/#LOCAL_STATIC_LIBRARIES := lapack/g' jni/Android.mk
sed -i 's/include $(BUILD_SHARED_LIBRARY)/#include $(BUILD_SHARED_LIBRARY)/g' jni/Android.mk
sed -i 's/minSdkVersion="10"/minSdkVersion="21"/g' AndroidManifest.xml
sed -i 's/gnustl_static/c++_static/g' jni/Application.mk
sed -i 's/armeabi armeabi-v7a/armeabi-v7a/g' jni/Application.mk
sed -i 's/android-10/android-21/g' project.properties

# build for android
/path/to/android-ndk-r22b/ndk-build
```

**拷贝`obj/local/armeabi-v7a/` 中的库到OpenBLAS 安装库目录 (e.g: OpenBlas/install/lib)**. kaldi会在该目录寻找 libf2c.a, liblapack.a, libclapack.a,libblas.a.



### 3. 编译openfst

```shell
export ANDROID_NDK=/path/to/android-ndk-r22b
export TOOLCHAIN=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64
export TARGET=armv7a-linux-androideabi
export API=21
export AR=$TOOLCHAIN/bin/llvm-ar
export CC=$TOOLCHAIN/bin/$TARGET$API-clang
export AS=$CC
export CXX=$TOOLCHAIN/bin/$TARGET$API-clang++
export LD=$TOOLCHAIN/bin/ld
export RANLIB=$TOOLCHAIN/bin/llvm-ranlib
export STRIP=$TOOLCHAIN/bin/llvm-strip
export PATH=${TOOLCHAIN}/bin:$PATH
wget -T 10 -t 1 http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.7.tar.gz 
tar -zxvf openfst-1.6.7.tar.gz
cd openfst-1.6.7/
./configure --prefix=`pwd` --enable-static --enable-shared --enable-far --enable-ngram-fsts --host=$TARGET LIBS="-ldl"
make -j 4
make install
cd ..
ln -s openfst-1.6.7 openfst
```

### 4. 编译kaldi

```shell
cd kaldi/tools
# 链接openfst安装目录
ln -s ../../openfst-1.6.7 openfst

export ANDROID_NDK=/path/to/android-ndk-r22b
export TOOLCHAIN=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64
export TARGET=armv7a-linux-androideabi
export API=21
export AR=$TOOLCHAIN/bin/llvm-ar
export CC=$TOOLCHAIN/bin/$TARGET$API-clang
export AS=$CC
export CXX=$TOOLCHAIN/bin/$TARGET$API-clang++
export LD=$TOOLCHAIN/bin/ld
export RANLIB=$TOOLCHAIN/bin/llvm-ranlib
export STRIP=$TOOLCHAIN/bin/llvm-strip
export PATH=${TOOLCHAIN}/bin:$PATH

cd kaldi/src
#You may want to compile Kaldi without debugging symbols.
#In this case, do:
sed -i 's/-g # -O0 -DKALDI_PARANOID/-O3 -DNDEBUG/g' kaldi.mk
# 删除test，在编译android的时候会运行test，在linux上会运行失败
rm -rf `find -name "*test.cc"`
sed -i 's/TESTFILES/#TESTFILES/g' matrix/Makefile
# 在config中不修改AR等，使用环境变量配置
sed -i 's/CXX="$HOST-$CXX"/#CXX="$HOST-$CXX"/g' configure
sed -i 's/AR="$HOST-$AR"/#AR="$HOST-$AR"/g' configure
sed -i 's/AS="$HOST-$AS"/#AS="$HOST-$AS"/g' configure
sed -i 's/RANLIB="$HOST-$RANLIB"/#RANLIB="$HOST-$RANLIB"/g' configure


./configure --static --android-incdir=/path/to/android-ndk-r22b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/ --host=$TARGET --openblas-root=/path/to/OpenBLAS/install --use-cuda=no
make depend -j4
make -j4
```

## ndk r22 64位编译

### 1. 编译android版本的openblas

```shell
export ANDROID_NDK=/path/to/android-ndk-r22b
export TOOLCHAIN=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64

# 64位
export TARGET=aarch64-linux-android
export API=21
make TARGET=ARMV8 ONLY_CBLAS=1 AR=$TOOLCHAIN/bin/llvm-ar RANLIB=$TOOLCHAIN/bin/llvm-ranlib CC=$TOOLCHAIN/bin/$TARGET$API-clang HOSTCC=gcc -j4
make install TARGET=ARMV8 NO_SHARED=1 PREFIX=`pwd`/install64
```

### 2. 编译CLAPACK

```shell
git clone https://github.com/simonlynen/android_libs.git

cd android_libs/lapack

# remove some compile instructions related to tests
sed -i 's/LOCAL_MODULE:= testlapack/#LOCAL_MODULE:= testlapack/g' jni/Android.mk
sed -i 's/LOCAL_SRC_FILES:= testclapack.cpp/#LOCAL_SRC_FILES:= testclapack.cpp/g' jni/Android.mk
sed -i 's/LOCAL_STATIC_LIBRARIES := lapack/#LOCAL_STATIC_LIBRARIES := lapack/g' jni/Android.mk
sed -i 's/include $(BUILD_SHARED_LIBRARY)/#include $(BUILD_SHARED_LIBRARY)/g' jni/Android.mk
sed -i 's/minSdkVersion="10"/minSdkVersion="21"/g' AndroidManifest.xml
sed -i 's/gnustl_static/c++_static/g' jni/Application.mk
sed -i 's/armeabi armeabi-v7a/arm64-v8a/g' jni/Application.mk
sed -i 's/android-10/android-21/g' project.properties

# build for android
/path/to/android-ndk-r22b/ndk-build
```

**拷贝`obj/local/armeabi-v7a/` 中的库到OpenBLAS 安装库目录 (e.g: OpenBlas/install/lib)**. kaldi会在该目录寻找 libf2c.a, liblapack.a, libclapack.a,libblas.a.



### 3. 编译openfst

```shell
export ANDROID_NDK=/path/to/android-ndk-r22b
export TOOLCHAIN=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64
export TARGET=aarch64-linux-android
export API=21
export AR=$TOOLCHAIN/bin/llvm-ar
export CC=$TOOLCHAIN/bin/$TARGET$API-clang
export AS=$CC
export CXX=$TOOLCHAIN/bin/$TARGET$API-clang++
export LD=$TOOLCHAIN/bin/ld
export RANLIB=$TOOLCHAIN/bin/llvm-ranlib
export STRIP=$TOOLCHAIN/bin/llvm-strip
wget -T 10 -t 1 http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.7.tar.gz 
tar -zxvf openfst-1.6.7.tar.gz
cd openfst-1.6.7/
./configure --prefix=`pwd` --enable-static --enable-shared --enable-far --enable-ngram-fsts --host=$TARGET LIBS="-ldl"
make -j 4
make install
cd ..
ln -s openfst-1.6.7 openfst
```

### 4. 编译kaldi

```shell
cd kaldi/tools
# 链接openfst安装目录
ln -s ../../openfst-1.6.7 openfst

export ANDROID_NDK=/path/to/android-ndk-r22b
export TOOLCHAIN=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64
export TARGET=aarch64-linux-android
export API=21
export AR=$TOOLCHAIN/bin/llvm-ar
export CC=$TOOLCHAIN/bin/$TARGET$API-clang
export AS=$CC
export CXX=$TOOLCHAIN/bin/$TARGET$API-clang++
export LD=$TOOLCHAIN/bin/ld
export RANLIB=$TOOLCHAIN/bin/llvm-ranlib
export STRIP=$TOOLCHAIN/bin/llvm-strip
export PATH=${TOOLCHAIN}/bin:$PATH
cd kaldi/src
#You may want to compile Kaldi without debugging symbols.
#In this case, do:
sed -i 's/-g # -O0 -DKALDI_PARANOID/-O3 -DNDEBUG/g' kaldi.mk
# 删除test，在编译android的时候会运行test，在linux上会运行失败
rm -rf `find -name "*test.cc"`
sed -i 's/TESTFILES/#TESTFILES/g' matrix/Makefile
# 在config中不修改AR等，使用环境变量配置
sed -i 's/CXX="$HOST-$CXX"/#CXX="$HOST-$CXX"/g' configure
sed -i 's/AR="$HOST-$AR"/#AR="$HOST-$AR"/g' configure
sed -i 's/AS="$HOST-$AS"/#AS="$HOST-$AS"/g' configure
sed -i 's/RANLIB="$HOST-$RANLIB"/#RANLIB="$HOST-$RANLIB"/g' configure

./configure --static --android-incdir=/path/to/android-ndk-r22b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/ --host=$TARGET --openblas-root=/path/to/OpenBLAS/install --use-cuda=no
make depend -j4
make -j4
```

## 参考

https://jcsilva.github.io/2017/03/18/compile-kaldi-android/

https://medium.com/swlh/compile-kaldi-for-64-bit-android-on-ubuntu-18-70967eb3a308

https://developer.android.com/ndk/guides/other_build_systems

https://github.com/xianyi/OpenBLAS/wiki/How-to-build-OpenBLAS-for-Android