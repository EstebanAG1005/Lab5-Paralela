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
        printf("Thread #%d, A[%d]=%.2f + B[%d]=%.2f => C[%d]=%.2f\n", i, i, A[i], i, B[i], i, C[i]);
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

    // Allocate y inicializar los vectores A y B en el host
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size); // Vector resultado en el host

    randomInit(h_A, numElements);
    randomInit(h_B, numElements);

    // // Imprimir los vectores A y B
    // // printf("Vector A inicializado:\n");
    // for (int i = 0; i < numElements; i++)
    // {
    //     printf("%.2f ", h_A[i]);
    // }
    // // printf("\n\nVector B inicializado:\n");
    // for (int i = 0; i < numElements; i++)
    // {
    //     printf("%.2f ", h_B[i]);
    // }
    // printf("\n\n");

    // Allocate los vectores en la memoria del device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copiar los vectores del host al device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Medir tiempo para la suma en GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Lanzar el kernel CUDA
    int threadsPerBlock = 1024; // Máximo común para muchas GPUs
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Esperar a que todos los threads terminen
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copiar el vector resultado de vuelta al host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Imprimir el vector resultado y la suma total
    float totalSum = 0.0f;
    // printf("Vector C (Resultado):\n");
    for (int i = 0; i < numElements; i++)
    {
        totalSum += h_C[i];
    }
    printf("\nTiempo de ejecución en la GPU: %f ms\n", milliseconds);
    printf("\nResultado de la suma de los elementos en C: %.2f\n", totalSum);

    // Liberar la memoria del device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Liberar la memoria del host
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
