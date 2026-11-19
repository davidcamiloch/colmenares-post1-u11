/*
 * vectorAddStream.cu - Suma de vectores con transferencia asíncrona
 * (cudaMemcpyAsync) y streams que solapan copia y cómputo por chunks.
 * Arquitectura de Computadores - Unidad 11 - Post-contenido 1.
 *
 * Uso:
 *   ./vectorAddStream          -> síncrono vs 4 streams
 *   ./vectorAddStream 8        -> síncrono vs 8 streams
 *   ./vectorAddStream sweep    -> barrido con 1, 2, 4, 8, 16 y 32 streams
 *
 * Compilar: nvcc -O2 -arch=sm_75 -o vectorAddStream src/vectorAddStream.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

#define N (1 << 24)        /* 16M elementos, igual que vectorAdd.cu */
#define BLOCK_SIZE 256
#define MAX_STREAMS 32
#define REPS 5             /* repeticiones por medición; se promedian */

__global__ void vectorAddChunk(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

/* Buffers compartidos por todas las mediciones */
static float *h_A, *h_B, *h_C, *h_ref;   /* host pinned + referencia */
static float *d_A, *d_B, *d_C;           /* dispositivo */
static cudaEvent_t evStart, evStop;

static int verify(void)
{
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(h_C[i] - h_ref[i]) > 1e-4f) errors++;
    return errors;
}

/* Baseline síncrono: H2D completo, un kernel, D2H completo. */
static float runSync(int *errors)
{
    size_t bytes = (size_t)N * sizeof(float);
    float total = 0.0f;
    for (int r = 0; r < REPS; r++) {
        memset(h_C, 0, bytes);
        CUDA_CHECK(cudaEventRecord(evStart));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
        vectorAddChunk<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
        total += ms;
    }
    *errors = verify();
    return total / REPS;
}

/* Versión con streams: cada stream copia su chunk, lo suma y lo devuelve. */
static float runStreams(int nstreams, int *errors)
{
    cudaStream_t streams[MAX_STREAMS];
    for (int s = 0; s < nstreams; s++)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    /* El último chunk absorbe el residuo si N no es divisible. */
    int chunk = N / nstreams;
    float total = 0.0f;
    for (int r = 0; r < REPS; r++) {
        memset(h_C, 0, (size_t)N * sizeof(float));
        CUDA_CHECK(cudaEventRecord(evStart));
        for (int s = 0; s < nstreams; s++) {
            size_t off = (size_t)s * chunk;
            int n = (s == nstreams - 1) ? N - s * chunk : chunk;
            size_t cb = (size_t)n * sizeof(float);
            int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            CUDA_CHECK(cudaMemcpyAsync(d_A + off, h_A + off, cb,
                                       cudaMemcpyHostToDevice, streams[s]));
            CUDA_CHECK(cudaMemcpyAsync(d_B + off, h_B + off, cb,
                                       cudaMemcpyHostToDevice, streams[s]));
            vectorAddChunk<<<grid, BLOCK_SIZE, 0, streams[s]>>>(
                d_A + off, d_B + off, d_C + off, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h_C + off, d_C + off, cb,
                                       cudaMemcpyDeviceToHost, streams[s]));
        }
        for (int s = 0; s < nstreams; s++)
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
        total += ms;
    }
    *errors = verify();
    for (int s = 0; s < nstreams; s++)
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    return total / REPS;
}

/* Ancho de banda H2D efectivo de un buffer (pinned o paginable). */
static double bandwidthH2D(const float *src)
{
    size_t bytes = (size_t)N * sizeof(float);
    CUDA_CHECK(cudaEventRecord(evStart));
    for (int r = 0; r < REPS; r++)
        CUDA_CHECK(cudaMemcpy(d_A, src, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
    return (double)bytes * REPS / (ms * 1.0e6);   /* GB/s */
}

/* Tiempo de solo kernel para los 16M elementos (referencia de cómputo). */
static float kernelOnly(void)
{
    CUDA_CHECK(cudaEventRecord(evStart));
    for (int r = 0; r < REPS; r++)
        vectorAddChunk<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
    return ms / REPS;
}

int main(int argc, char **argv)
{
    size_t bytes = (size_t)N * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | copy engines (asyncEngineCount): %d | "
           "concurrentKernels: %d\n",
           prop.name, prop.asyncEngineCount, prop.concurrentKernels);

    /* Memoria host pinned (page-locked): obligatoria para que
     * cudaMemcpyAsync sea realmente asíncrona respecto al host. */
    CUDA_CHECK(cudaHostAlloc(&h_A, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_B, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_C, bytes, cudaHostAllocDefault));
    h_ref = (float *)malloc(bytes);
    if (!h_ref) { fprintf(stderr, "Sin memoria en host\n"); return 1; }
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
        h_ref[i] = h_A[i] + h_B[i];   /* referencia de CPU */
    }

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    /* Calentamiento: carga perezosa del módulo y primeras copias. */
    int e;
    runStreams(2, &e);
    runSync(&e);

    double bwPinned = bandwidthH2D(h_A);
    double bwPageable = bandwidthH2D(h_ref);
    float kms = kernelOnly();
    printf("Ancho de banda H2D: pinned %.2f GB/s | paginable %.2f GB/s\n",
           bwPinned, bwPageable);
    printf("Volumen transferido: %.0f MB | kernel solo: %.3f ms\n",
           3.0 * bytes / 1.0e6, kms);

    int errSync;
    float syncMs = runSync(&errSync);
    printf("Sincrono (H2D + kernel + D2H completos): %.2f ms | Errores: %d\n",
           syncMs, errSync);

    int list[] = {1, 2, 4, 8, 16, 32};
    int count = 6;
    int single = 4;
    if (argc > 1 && strcmp(argv[1], "sweep") != 0) {
        single = atoi(argv[1]);
        if (single < 1 || single > MAX_STREAMS) single = 4;
    }
    if (argc > 1 && strcmp(argv[1], "sweep") == 0) {
        for (int i = 0; i < count; i++) {
            int errS;
            float ms = runStreams(list[i], &errS);
            printf("Con streams (%2d, solapado): %.2f ms | Reduccion: %5.1f%%"
                   " | Errores: %d\n", list[i], ms,
                   100.0 * (syncMs - ms) / syncMs, errS);
        }
    } else {
        int errS;
        float asyncMs = runStreams(single, &errS);
        printf("Con streams (%d, solapado): %.2f ms | Errores: %d\n", single,
               asyncMs, errS);
        printf("Reduccion: %.1f%%\n", 100.0 * (syncMs - asyncMs) / syncMs);
    }

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    free(h_ref);
    return 0;
}
