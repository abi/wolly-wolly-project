TARGET  := match
SRCS    := $(shell find . -name '*.cpp')
OBJS    := ${SRCS:.cpp=.o} 
DEPS    := ${SRCS:.cpp=.dep} 

CXX = g++
CCFLAGS = -g -I/usr/local/Cellar/opencv/2.3.1a/include/opencv -I/usr/local/Cellar/opencv/2.3.1a/include -I/usr/local/Cellar/boost/1.47.0/include/boost  -I/usr/local/Cellar/boost/1.47.0/include
LDFLAGS = -L/usr/local/Cellar/opencv/2.3.1a/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann -L/usr/local/Cellar/boost/1.47.0/lib -lboost_chrono-mt -lboost_date_time-mt -lboost_exception-mt -lboost_filesystem-mt -lboost_graph-mt -lboost_iostreams-mt -lboost_math_c99-mt -lboost_math_c99f-mt -lboost_math_c99l-mt -lboost_math_tr1-mt -lboost_math_tr1f-mt -lboost_math_tr1l-mt -lboost_prg_exec_monitor-mt -lboost_program_options-mt -lboost_random-mt -lboost_regex-mt -lboost_serialization-mt -lboost_signals-mt -lboost_system-mt -lboost_test_exec_monitor-mt -lboost_thread-mt -lboost_unit_test_framework-mt -lboost_wave-mt -lboost_wserialization-mt
LIBS    =
# TODO: Are the libraries already in the path that we want?

.PHONY: all clean distclean 
all:: ${TARGET} 

${TARGET}: ${OBJS} 
	${CXX} ${LDFLAGS} -o $@ $^ ${LIBS} 

${OBJS}: %.o: %.cpp %.dep 
	${CXX} ${CCFLAGS} -o $@ -c $< 

${DEPS}: %.dep: %.cpp Makefile
	${CXX} ${CCFLAGS} -MM $< > $@

abi:
	${CXX} ${LDFLAGS} main.c ${LIBS}

imread:
	${CXX} ${CCFLAGS} ${LDFLAGS} read.cpp ${LIBS}

# ${TARGET}
clean:: 
	-rm -f *~ *.o *.dep ${TARGET}

distclean:: clean
