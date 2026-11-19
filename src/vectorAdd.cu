/*
 * vectorAdd.cu - Suma de vectores C = A + B en CPU y GPU.
 * Arquitectura de Computadores - Unidad 11 - Post-contenido 1.
 *
 * Mide para N = 1M, 4M y 16M elementos:
 *   - tiempo de CPU (referencia secuencial),
 *   - tiempo del kernel en GPU (cudaEvent, sin transferencias),
 *   - tiempo total en GPU (H2D + kernel + D2H, cudaEvent).
 *
 * Compilar: nvcc -O2 -arch=sm_75 -o vectorAdd src/vectorAdd.cu
 * Ejecutar: ./vectorAdd
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/* Aborta con mensaje claro ante cualquier error de la API de CUDA. */
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err_ = (call);                                    \
        if (err_ != cudaSuccess) {                                    \
            fprintf(stderr, "Error CUDA %s:%d: %s\n", __FILE__,      \
                    __LINE__, cudaGetErrorString(err_));              \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

#define N_MAX (1 << 24)   /* 16M elementos (16 777 216) */
#define BLOCK_SIZE 256
#define REPS 10           /* repeticiones para promediar el kernel */

/* Kernel CUDA: cada thread calcula un elemento de C. */
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

/* Suma de vectores en CPU (referencia). */
static void vectorAddCPU(const float *A, const float *B, float *C, int n)
{
    for (int i = 0; i < n; i++) C[i] = A[i] + B[i];
}

/* Reloj de pared en milisegundos (clock() mide tiempo de CPU del proceso). */
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static void run(int n)
{
    size_t bytes = (size_t)n * sizeof(float);

    /* Memoria en host */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_cpu = (float *)malloc(bytes);
    float *h_C_gpu = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Sin memoria en host\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    /* --- Benchmark CPU --- */
    double t0 = now_ms();
    vectorAddCPU(h_A, h_B, h_C_cpu, n);
    double cpu_ms = now_ms() - t0;

    /* --- Benchmark GPU --- */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Tiempo total: H2D + kernel + D2H */
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    /* Tiempo de kernel: promedio de REPS lanzamientos ya en caliente */
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < REPS; r++)
        vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    kernel_ms /= REPS;

    /* Verificación contra la referencia de CPU */
    int errors = 0;
    for (int i = 0; i < n; i++)
        if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > 1e-4f) errors++;

    /* 12 bytes por elemento: 2 lecturas y 1 escritura de float */
    double gbps = 12.0 * n / (kernel_ms * 1.0e6);
    printf("N = %8d | CPU: %8.2f ms | GPU kernel: %6.3f ms (%5.1f GB/s) | "
           "GPU total: %7.2f ms | Errores: %d\n",
           n, cpu_ms, kernel_ms, gbps, total_ms, errors);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
}

int main(void)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | CC %d.%d | SMs: %d\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount);

    /* Calentamiento: crea el contexto y carga el módulo (lazy loading). */
    CUDA_CHECK(cudaFree(0));
    run(1 << 10);

    run(1 << 20);   /*  1M */
    run(1 << 22);   /*  4M */
    run(N_MAX);     /* 16M */
    return 0;
}
