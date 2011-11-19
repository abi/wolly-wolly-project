TARGET  := match
SRCS    := $(shell find . -name '*.cpp')
OBJS    := ${SRCS:.cpp=.o} 
DEPS    := ${SRCS:.cpp=.dep} 

CXX = g++
CCFLAGS = -arch i386 -g -I./ -DSTANDALONE_DEBUG
# -I/usr/local/Cellar/opencv/2.2/include/opencv/
LDFLAGS = -arch i386
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

# ${TARGET}
clean:: 
	-rm -f *~ *.o *.dep ${TARGET}

distclean:: clean