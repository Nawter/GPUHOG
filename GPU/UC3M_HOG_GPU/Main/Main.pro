TEMPLATE = app
QT = gui
CONFIG += debug console
ONFIG += 32bit

CONFIG(32bit) {
    TARGET = 32bit_binary
    QMAKE_CXXFLAGS += -m32
    LIBS += -lcudart -L/usr/local/cuda/lib -lrt
}

HEADERS =
SOURCES = \
    Main.cpp

INCLUDEPATH += ../UC3M_HOG_GPU
LIBS += -lUC3M_HOG_GPU -L../lib -lcudart -L/usr/local/cuda/lib64

DESTDIR =../bin
