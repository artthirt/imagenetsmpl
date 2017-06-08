INCLUDEPATH += $$PWD

DEFINES += _USE_GPU

HEADERS += \
    $$PWD/gpumat.h \
    $$PWD/helper_gpu.h \
    $$PWD/cuda_common.h \
    $$PWD/convnn_gpu.h \
    $$PWD/cu/common_devices.h \
    $$PWD/gpu_mlp.h \
    $$PWD/cuda_types.h \
    $$PWD/convnn2_gpu.h \
    $$PWD/qt_work_mat.h

SOURCES += \
    $$PWD/gpumat.cpp \
    $$PWD/helper_gpu.cpp \
    $$PWD/convnn_gpu.cpp \
    $$PWD/gpu_mlp.cpp \
    $$PWD/cuda_common.cpp \
    $$PWD/cuda_types.cpp \
    $$PWD/convnn2_gpu.cpp \
    $$PWD/qt_work_mat.cpp

CUDA_SOURCES += $$PWD/cu/cuda_arithm.cu \
                $$PWD/cu/cuda_conv.cu \
                $$PWD/cu/cuda_conv2.cu

win32{
    CUDA_DIR					= "c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/"
    MSVCRT_LINK_FLAG_DEBUG		= "/MDd"
    MSVCRT_LINK_FLAG_RELEASE	= "/MD"
    SYSTEM_NAME					= Win64         # Depending on your system either 'Win32', 'x64', or 'Win64'
# Path to cuda toolkit install
# Path to header and libs files
    INCLUDEPATH  += $$CUDA_DIR/include
    QMAKE_LIBDIR += $$CUDA_DIR/lib/x64     # Note I'm using a 64 bits Operating system
}else{
    CUDA_DIR					= /usr/
    SYSTEM_NAME					= unix         # Depending on your system either 'Win32', 'x64', or 'Win64'
    CUDA_OBJECTS_DIR			= ./
}

SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_21           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math

# Add the necessary libraries
CUDA_LIBS = -lcuda -lcudart

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so')
LIBS += $$CUDA_LIBS

CONFIG(debug, debug|release){
    CUDA_OBJECTS_DIR = tmp/debug/cuda
}else{
    CUDA_OBJECTS_DIR = tmp/release/cuda
}

CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    #-D_DEBUG

    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC \
                    $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}

    win32{
        cuda_d.commands += --compile -cudart static -g -DWIN32 -D_MBCS \
                    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
                    -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG
    }else{
        cuda_d.commands += -ccbin gcc-5\
    }

    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS \
                    --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH\
                    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}

    win32{
        cuda.commands +=  --compile -cudart static -DWIN32 -D_MBCS \
                    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
                    -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE
    }else{
        cuda.commands += -ccbin gcc-5\
    }

    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}


