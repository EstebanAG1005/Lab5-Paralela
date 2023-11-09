#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Función para inicializar un vector con valores aleatorios entre 0 y 1
void initializeVector(float *vector, int n)
{
    for (int i = 0; i < n; i++)
    {
        vector[i] = (float)rand() / RAND_MAX;
    }
}

// Función para sumar dos vectores
void vectorAdditionCPU(float *a, float *b, float *result, int n)
{
    for (int i = 0; i < n; i++)
    {
        result[i] = a[i] + b[i];
        printf("Elemento #%d, A[%d]=%.2f + B[%d]=%.2f => C[%d]=%.2f\n", i, i, a[i], i, b[i], i, result[i]);
    }
}

int main()
{
    int numElements = 1000000; // El número de elementos en los vectores
    srand((unsigned int)time(NULL));

    // Vectores en la CPU
    float *h_a = (float *)malloc(numElements * sizeof(float));
    float *h_b = (float *)malloc(numElements * sizeof(float));
    float *h_result = (float *)malloc(numElements * sizeof(float));

    // Inicializar los vectores con valores aleatorios
    initializeVector(h_a, numElements);
    initializeVector(h_b, numElements);

    // Medir el tiempo de inicio
    clock_t start_time = clock();

    // Realizar la suma en la CPU
    vectorAdditionCPU(h_a, h_b, h_result, numElements);

    // Medir el tiempo de finalización
    clock_t end_time = clock();
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0;

    // Imprimir el tiempo de ejecución
    printf("Tiempo de ejecución en la CPU: %f ms\n", execution_time);

    // Calcular e imprimir la suma total de los elementos en C
    float totalSum = 0.0f;
    for (int i = 0; i < numElements; i++)
    {
        printf("%.2f ", h_result[i]);
        totalSum += h_result[i];
    }
    printf("Resultado de la suma de los elementos en C: %.2f\n", totalSum);

    // Liberar memoria en la CPU
    free(h_a);
    free(h_b);
    free(h_result);

    return 0;
}
