#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// CUDA Kernel para sumar dos vectores
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

// Función para inicializar los vectores con valores aleatorios entre 0 y 1
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = rand() / (float)RAND_MAX;
    }
}

int main(void)
{
    int numElements = 1000000; // El número de elementos en los vectores
    size_t size = numElements * sizeof(float);
    srand(time(NULL)); // Inicializar la semilla de números aleatorios

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size); // Vector resultado en el host

    randomInit(h_A, numElements);
    randomInit(h_B, numElements);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    float totalMilliseconds = 0;

    // Ejecutar el kernel 10 veces y sumar el tiempo total
    for (int i = 0; i < 10; ++i)
    {
        cudaEventRecord(start);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalMilliseconds += milliseconds;
    }

    // Calcular el tiempo promedio de ejecución
    float averageMilliseconds = totalMilliseconds / 10;
    printf("Tiempo promedio de cálculo del kernel en la GPU: %f ms\n", averageMilliseconds);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    float totalSum = 0.0f;
    for (int i = 0; i < numElements; i++)
    {
        totalSum += h_C[i];
    }

    printf("\nResultado de la suma de los elementos en C: %.2f\n", totalSum);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
