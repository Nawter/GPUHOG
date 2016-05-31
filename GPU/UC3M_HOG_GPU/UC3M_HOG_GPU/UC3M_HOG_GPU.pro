TEMPLATE = lib
#CONFIG += debug console

QMAKE_CXXFLAGS += -fPIC -ffast-math -fomit-frame-pointer 
QMAKE_CFLAGS += -fPIC -ffast-math -fomit-frame-pointer

HEADERS = \
    TrainSVM.h \    
    TestPadImage.h \
    SkipSupperssion.h \
    CalculateGradients.h \
    ComputeBlocks.h \
    UseConversions.h \
    Global.h \
    Timer.h \
    DumpDetections.h \
    HOG.h\
    LocalParameters.h \
    GlobalParameters.h \ 
    HOGPlanner.h
SOURCES = \
    LocalParameters.cpp \
    GlobalParameters.cpp \
    HOGPlanner.cpp
#    Timer.cu \
#    TrainSVM.cu \
#    TestPadImage.cu \
#    SkipSupperssion.cu \
#    HOG.cu \
#    CalculateGradients.cu \
#    ComputeBlocks.cu \
#    UseConversions.cu \
#    DumpDetections.cu



CUSOURCES += Timer.cu \
    TrainSVM.cu \ 
    TestPadImage.cu \
    SkipSupperssion.cu \
    HOG.cu \
    CalculateGradients.cu \
    ComputeBlocks.cu \
    UseConversions.cu \
    DumpDetections.cu

CUDA_SDK_PATH = /usr/local/cuda
LIBS += -lcudart -L/usr/local/cuda/lib64 -lrt

QMAKE_CUC = /usr/local/cuda/bin/nvcc
cu.name = Cuda ${QMAKE_FILE_IN}
cu.input = CUSOURCES
cu.CONFIG += no_link
cu.variable_out = OBJECTS

INCLUDEPATH += /usr/local/cuda/include 
QMAKE_CUFLAGS += $$QMAKE_CFLAGS

QMAKE_CUEXTRAFLAGS += -arch=sm_30 --ptxas-options=-v -Xcompiler -fPIC -Xcompiler -rdynamic -lineinfo -Xcompiler $$join(QMAKE_CUFLAGS, ",")
#QMAKE_CUEXTRAFLAGS += -arch=sm_11 -Xcompiler -fPIC -Xcompiler $$join(QMAKE_CUFLAGS, ",")
QMAKE_CUEXTRAFLAGS += $(DEFINES) $(INCPATH) $$join(QMAKE_COMPILER_DEFINES, " -D", -D)
QMAKE_CUEXTRAFLAGS += -c

cu.commands = $$QMAKE_CUC $$QMAKE_CUEXTRAFLAGS -o ${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ} ${QMAKE_FILE_NAME}$$escape_expand(\n\t)
cu.output = ${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ}
silent:cu.commands = @echo nvcc ${QMAKE_FILE_IN} && $$cu.commands
QMAKE_EXTRA_COMPILERS += cu

build_pass|isEmpty(BUILDS):cuclean.depends = compiler_cu_clean
else:cuclean.CONFIG += recursive
QMAKE_EXTRA_TARGETS += cuclean

