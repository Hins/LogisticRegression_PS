CPPFLAGS=-g -Wall 

BOOST_HOME=/home/xtpan/

LIBS= -L /usr/include

INCLUDE_PATH=include

DIR_SRC = src
SRC = $(wildcard ${DIR_SRC}/*.cpp)
OBJS = $(patsubst %.cpp, %.o, ${SRC})

CPLUS_INCLUDE_PATH=${BOOST_HOME}/include
export CPLUS_INCLUDE_PATH

CXX=mpic++

.PHONY : clean all

all: ${OBJS} lr

%.o: %.cpp
	$(CXX) -c $(CPPFLAGS) ${LIBS} -I$(INCLUDE_PATH) $^ -o $@
lr: lr.cpp  ${OBJS}
	$(CXX) $(CPPFLAGS) -I$(INCLUDE_PATH) $^  ${LIBS} -o $@

clean:
	rm -rf  ${DIR_SRC}/*.o  lr
