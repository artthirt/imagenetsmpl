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

HEADERS += \
    imnetsmpl.h \
    imreader.h \
    #qt_work_mat.h

CONFIG(debug, debug|release){
    TMP = debug
}else{
    TMP = release
}

win32{

    QMAKE_CXXFLAGS += /openmp

    CONFIG(debug, debug|release){
        libs = -lopencv_core320d -lopencv_highgui320d -lopencv_imgproc320d -lopencv_imgcodecs320d
    }else{
        libs = -lopencv_core320 -lopencv_highgui320 -lopencv_imgproc320 -lopencv_imgcodecs320
    }

    INCLUDEPATH += $(OPENCV3_DIR)/include
    LIBS += -L$(OPENCV3_DIR)/x64/vc14/lib $$libs
}else{
    QMAKE_CXXFLAGS += -fopenmp
    LIBS += -l:libopencv_core.so.3.2 -l:libopencv_highgui.so.3.2 -l:libopencv_imgproc.so.3.2 -l:libopencv_imgcodecs.so.3.2 -ltbb -lgomp
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

include(ct/ct.pri)
#include(gpu/gpu.pri)

UI_DIR += tmp/$$TMP/ui
OBJECTS_DIR += tmp/$$TMP/obj
RCC_DIR += tmp/$$TMP/rcc
MOC_DIR += tmp/$$TMP/moc
