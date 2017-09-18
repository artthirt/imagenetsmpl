QT += core
QT -= gui

CONFIG += c++11

TARGET = imagenetsmpl
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    imnetsmpl.cpp \
    imreader.cpp \
   # qt_work_mat.cpp
    imnetsmplgpu.cpp \
    imnet_list.cpp

HEADERS += \
    imnetsmpl.h \
    imreader.h \
    #qt_work_mat.h
    imnetsmplgpu.h \
    imnet_list.h

CONFIG(debug, debug|release){
    TMP = debug
}else{
    TMP = release
}

win32{

    isEmpty(OPENCV_VER){
        OPENCV_VER = 310
    }

    QMAKE_CXXFLAGS += /openmp

    VER = $$OPENCV_VER

    CONFIG(debug, debug|release){
        VER = $$OPENCV_VER"d"
        LIBS = -lopencv_core$$VER -lopencv_highgui$$VER -lopencv_imgproc$$VER -lopencv_imgcodecs$$VER
    }else{
        LIBS = -lopencv_core$$VER -lopencv_highgui$$VER -lopencv_imgproc$$VER -lopencv_imgcodecs$$VER
    }

    INCLUDEPATH += $$OPENCV3_DIR/include
    LIBS += -L$$OPENCV3_DIR/x64/vc14/lib $$libs
}else{
    QMAKE_CXXFLAGS += -fopenmp
    LIBS += -l:libopencv_core.so -l:libopencv_highgui.so -l:libopencv_imgproc.so -l:libopencv_imgcodecs.so -ltbb -lgomp
}

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

include(ml_algorithms/ct/ct.pri)
include(ml_algorithms/gpu/gpu.pri)

UI_DIR += tmp/$$TMP/ui
OBJECTS_DIR += tmp/$$TMP/obj
RCC_DIR += tmp/$$TMP/rcc
MOC_DIR += tmp/$$TMP/moc
