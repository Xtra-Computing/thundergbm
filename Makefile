CCFLAGS	  := -Wall -std=c++11
NVCCFLAGS := -arch=sm_60 -lrt -Wno-deprecated-gpu-targets -dc -std=c++11
LASTFLAG  := -Wno-deprecated-gpu-targets
LDFLAGS   := -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -lcuda -lcudadevrt -lcudart -lcublas -lpthread -lcusparse
NVCC	  := /usr/local/cuda/bin/nvcc
DISABLEW  := -Xnvlink -w
CXX = g++

ODIR = bin
exe_name = gbdt
release_bin := $(ODIR)/release/$(exe_name)
debug_bin := $(ODIR)/debug/$(exe_name)
$(shell mkdir -p $(ODIR)/release)
$(shell mkdir -p $(ODIR)/debug)
$(shell mkdir -p $(ODIR)/$(obj_folder))

FILES = $(shell find ./ -name '*.cpp' -or -name '*.cu')
SOURCE = $(notdir $(FILES))             #remove directory
OBJS = $(patsubst %.cpp, %.o,$(SOURCE:.cpp=.o)) #replace .cpp to .o
OBJ = $(patsubst %.cu, %.o,$(OBJS:.cu=.o))      #replace .cu to .o

$(release_bin): $(OBJ)
	$(NVCC) $(LASTFLAG) $(LDFLAGS) $(DISABLEW) -o $(release_bin) $(OBJ)
$(debug_bin): $(OBJ)
	$(NVCC) $(LASTFLAG) $(LDFLAGS) $(DISABLEW) -o $(debug_bin) $(OBJ)

.PHONY: release
.PHONY: debug

release: CCFLAGS += -O2
release: NVCCFLAGS += -O2
release: LASTFLAG += -O2
release: $(release_bin)

debug: CCFLAGS += -g
debug: NVCCFLAGS += -G -g
debug: LASTFLAG += -G -g
debug: $(debug_bin)

#compile main 
%.o: %.c* */*h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<
				
#compile files under Host folder
%.o: Host/%.c* Host/*.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<
%.o: Host/*/%.c* Host/*/*.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<

%.o: Device/%.c* Device/*.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<
%.o: Device/*/%.c* Device/*/*.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<

%.o: DeviceHost/%.c* DeviceHost/*.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<

%.o: DeviceHost/*/%.c* DeviceHost/*/*.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<
	
%.o: SharedUtility/%.c* SharedUtility/*.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<
	
%.o: SharedUtility/*/%.c* SharedUtility/*/*.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<

all: release

.PHONY:clean

clean:
	rm -f *.o *.txt bin/*.bin bin/result.txt bin/release/* bin/debug/*
