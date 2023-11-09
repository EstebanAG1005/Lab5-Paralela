CC=gcc
NVCC=nvcc
CFLAGS=-I.
CUDA_PATH=/usr/local/cuda
CUDA_INCLUDEPATH=$(CUDA_PATH)/include
CUDA_LIBPATH=$(CUDA_PATH)/lib64

# Targets
all: VectorAdd VectorAdd_speedup VectorAdd_seq VectorAdd_Seq_Speed

VectorAdd: VectorAdd.cu
	$(NVCC) -o $@ $< -I$(CUDA_INCLUDEPATH) -L$(CUDA_LIBPATH) -lcudart

VectorAdd_speedup: VectorAdd_speedup.cu
	$(NVCC) -o $@ $< -I$(CUDA_INCLUDEPATH) -L$(CUDA_LIBPATH) -lcudart

VectorAdd_seq: VectorAdd_seq.c
	$(CC) -o $@ $< $(CFLAGS)

VectorAdd_Seq_Speed: VectorAdd_Seq_Speed.c
	$(CC) -o $@ $< $(CFLAGS)

clean:
	rm -f VectorAdd VectorAdd_speedup VectorAdd_seq VectorAdd_Seq_Speed
