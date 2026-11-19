/*
 * matMul.cu - Multiplicación de matrices C = A x B (N x N, float).
 * Arquitectura de Computadores - Unidad 11 - Post-contenido 1.
 *
 * Compara tres versiones para N = 512 y N = 1024:
 *   - CPU secuencial (referencia, acumulación en double),
 *   - kernel naive (cada thread lee fila y columna de memoria global),
 *   - kernel con tiling en shared memory (TILE = 16).
 *
 * Compilar: nvcc -O2 -arch=sm_75 -o matMul src/matMul.cu
 * Ejecutar: ./matMul
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err_ = (call);                                    \
        if (err_ != cudaSuccess) {                                    \
            fprintf(stderr, "Error CUDA %s:%d: %s\n", __FILE__,      \
                    __LINE__, cudaGetErrorString(err_));              \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

#define TILE 16
#define REPS 20   /* lanzamientos por medición; se reporta el promedio */

/* Kernel naive: 2N lecturas de memoria global por cada elemento de C. */
__global__ void matMulNaive(const float *A, const float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

/* Kernel con tiling: cada bloque carga tiles de TILE x TILE en shared
 * memory y los reutiliza TILE veces, lo que divide entre TILE las
 * lecturas de memoria global. */
__global__ void matMulTiled(const float *A, const float *B, float *C, int N)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        /* Cargar tile de A (con relleno de ceros en los bordes) */
        if (row < N && t * TILE + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        /* Cargar tile de B */
        if (col < N && t * TILE + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();   /* tile completo antes de usarlo */

        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();   /* nadie sobrescribe el tile mientras se usa */
    }

    if (row < N && col < N) C[row * N + col] = sum;
}

/* Referencia en CPU con orden i-k-j y acumulación en double. */
static void matMulCPU(const float *A, const float *B, float *C, int N)
{
    double *acc = (double *)calloc((size_t)N, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) acc[j] = 0.0;
        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];
            const float *b = &B[k * N];
            for (int j = 0; j < N; j++) acc[j] += a * b[j];
        }
        for (int j = 0; j < N; j++) C[i * N + j] = (float)acc[j];
    }
    free(acc);
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

/* Error absoluto máximo entre la salida de GPU y la referencia. */
static float maxAbsError(const float *ref, const float *got, int n)
{
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(ref[i] - got[i]);
        if (e > m) m = e;
    }
    return m;
}

/* Mide el promedio de REPS lanzamientos de un kernel con cudaEvent. */
typedef void (*kernel_t)(const float *, const float *, float *, int);

static float timeKernel(kernel_t k, dim3 grid, dim3 block, const float *d_A,
                        const float *d_B, float *d_C, int N)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    k<<<grid, block>>>(d_A, d_B, d_C, N);          /* calentamiento */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < REPS; r++) k<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / REPS;
}

static void run(int N)
{
    size_t elems = (size_t)N * N;
    size_t bytes = elems * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_cpu = (float *)malloc(bytes);
    float *h_C_gpu = (float *)malloc(bytes);
    srand(2026);
    for (size_t i = 0; i < elems; i++) {
        h_A[i] = 2.0f * rand() / RAND_MAX - 1.0f;   /* [-1, 1] */
        h_B[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    }

    double t0 = now_ms();
    matMulCPU(h_A, h_B, h_C_cpu, N);
    double cpu_ms = now_ms() - t0;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Grid 2D de tiles */
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    float naive_ms = timeKernel(matMulNaive, grid, block, d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    float errNaive = maxAbsError(h_C_cpu, h_C_gpu, (int)elems);

    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    float tiled_ms = timeKernel(matMulTiled, grid, block, d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    float errTiled = maxAbsError(h_C_cpu, h_C_gpu, (int)elems);

    double gflops = 2.0 * N * (double)N * N / (tiled_ms * 1.0e6);
    printf("N = %4d | CPU: %9.2f ms | naive: %7.3f ms | tiled: %7.3f ms"
           " | speedup tiled/naive: %.2fx | tiled vs CPU: %.0fx\n",
           N, cpu_ms, naive_ms, tiled_ms, naive_ms / tiled_ms,
           cpu_ms / tiled_ms);
    printf("         error max naive: %.2e | error max tiled: %.2e | "
           "tiled: %.1f GFLOP/s | %s\n",
           errNaive, errTiled, gflops,
           (errNaive < 1e-3f && errTiled < 1e-3f) ? "OK (< 1e-3)" : "FALLA");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
}

int main(void)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | TILE = %d | shared por bloque: %zu bytes\n", prop.name,
           TILE, 2 * TILE * TILE * sizeof(float));
    CUDA_CHECK(cudaFree(0));   /* crea el contexto antes de medir */

    run(512);
    run(1024);
    return 0;
}
