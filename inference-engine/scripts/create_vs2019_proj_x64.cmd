@echo off

pushd ..\..
if not exist "vs2019x64Debug" (
               mkdir "vs2019x64withDebug"
)
if not exist "vs2019x64Release" (
               mkdir "vs2019x64withRelease"
)

cmake -E chdir "vs2019x64withDebug" cmake -G "Visual Studio 16 2019" -A x64^
    -DCMAKE_BUILD_TYPE=Debug ^
    -DOS_FOLDER=ON ^
    -DENABLE_MYRIAD=OFF ^
    -DENABLE_VPU=OFF ^
    -DENABLE_GNA=ON ^
    -DENABLE_CLDNN=OFF ^
    -DENABLE_TESTS=OFF ^
    -DENABLE_BEH_TESTS=OFF ^
    -DENABLE_FUNCTIONAL_TESTS=OFF ^
    -DENABLE_PYTHON=ON ^
    -DENABLE_OPENCV=OFF ^
    -DENABLE_MKL_DNN=ON ^
    -DVERBOSE_BUILD=OFF ^
    -DENABLE_MODELS=OFF ^
    -DENABLE_VALIDATION_SET=OFF ^
    -DENABLE_PRIVATE_MODELS=OFF ^
    -DENABLE_CLANG_FORMAT=OFF ^
    -DGNA_LIBRARY_VERSION=GNA2 ^
    ..

cmake -E chdir "vs2019x64withRelease" cmake -G "Visual Studio 16 2019" -A x64^
    -DCMAKE_BUILD_TYPE=Release ^
    -DOS_FOLDER=ON ^
    -DENABLE_MYRIAD=OFF ^
    -DENABLE_VPU=OFF ^
    -DENABLE_GNA=ON ^
    -DENABLE_CLDNN=OFF ^
    -DENABLE_TESTS=OFF ^
    -DENABLE_BEH_TESTS=OFF ^
    -DENABLE_FUNCTIONAL_TESTS=OFF ^
    -DENABLE_PYTHON=ON ^
    -DENABLE_OPENCV=OFF ^
    -DENABLE_MKL_DNN=ON ^
    -DVERBOSE_BUILD=OFF ^
    -DENABLE_MODELS=OFF ^
    -DENABLE_VALIDATION_SET=OFF ^
    -DENABLE_PRIVATE_MODELS=OFF ^
    -DENABLE_CLANG_FORMAT=OFF ^
    -DGNA_LIBRARY_VERSION=GNA2 ^
    ..

popd
pause
