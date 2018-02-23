QT += core
QT -= gui

DESTDIR = ../

CONFIG += c++11

TARGET = imagenetsmpl
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    imnetsmpl.cpp \
    imreader.cpp \
   # qt_work_mat.cpp
    imnet_list.cpp

HEADERS += \
    imnetsmpl.h \
    imreader.h \
    #qt_work_mat.h
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
    LIBS += -l:libopencv_core.so -l:libopencv_highgui.so -l:libopencv_imgproc.so -l:libopencv_imgcodecs.so -ltbb -lgomp -lstdc++fs
}

include(../ml_algorithms/ct/ct.pri)

isEmpty(NOGPU){
    message("use gpu")

    SOURCES += imnetsmplgpu.cpp
    HEADERS += imnetsmplgpu.h

    include(../ml_algorithms/gpu/gpu_export.pri)
}else{
    message("don't use gpu")
}

UI_DIR += tmp/$$TMP/ui
OBJECTS_DIR += tmp/$$TMP/obj
RCC_DIR += tmp/$$TMP/rcc
MOC_DIR += tmp/$$TMP/moc

RESOURCES += \
    resource.qrc
