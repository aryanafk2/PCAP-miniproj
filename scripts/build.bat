@echo off
setlocal
cd /d "%~dp0.."
if not exist build mkdir build
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
if errorlevel 1 exit /b 1
cl /nologo /O2 /EHsc /std:c++17 src\sequential\main.cpp /Fo:build\sequential.obj /Fe:build\sequential.exe
if errorlevel 1 exit /b 1
cl /nologo /O2 /EHsc /std:c++17 /openmp src\openmp\main.cpp /Fo:build\openmp.obj /Fe:build\openmp.exe
if errorlevel 1 exit /b 1
cl /nologo /O2 /EHsc /std:c++17 /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" src\mpi\main.cpp /Fo:build\mpi.obj /Fe:build\mpi.exe /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib
if errorlevel 1 exit /b 1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -O3 -std=c++17 -arch=sm_120 src\cuda\main.cu -o build\cuda.exe
exit /b %errorlevel%
