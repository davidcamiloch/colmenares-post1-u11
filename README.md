# Post-Contenido 1 — CUDA Benchmark CPU vs GPU

Arquitectura de Computadores — Unidad 11 · David Camilo Colmenares Hernández · Ingeniería de Sistemas, UFPS

El repositorio reúne las dos partes del laboratorio. La Parte 1 implementa y mide tres programas CUDA en C: suma de vectores; multiplicación de matrices con tiling en shared memory; suma de vectores con streams y `cudaMemcpyAsync`. La Parte 2 está en [`docs/analisis-comparativo.md`](docs/analisis-comparativo.md) y compara GPU, TPU, NPU y FPGA para cargas de IA.

## Estructura

```
colmenares-post1-u11/
├── README.md
├── src/
│   ├── vectorAdd.cu          # CPU vs GPU kernel vs GPU total; N = 1M, 4M y 16M
│   ├── matMul.cu             # CPU vs naive vs tiled (TILE = 16); N = 512 y 1024
│   └── vectorAddStream.cu    # síncrono vs streams; barrido de 1 a 32 streams
├── capturas/                 # 10 capturas de Colab (2 por checkpoint)
└── docs/
    └── analisis-comparativo.md   # Parte 2
```

## 1. Entorno de ejecución

El estudiante no dispone de GPU NVIDIA física; el laboratorio se ejecutó en Google Colab con el acelerador GPU T4, como permite la guía.

| Elemento | Valor medido |
|---|---|
| GPU | NVIDIA Tesla T4 (Turing; compute capability 7.5; 40 SM; 15 360 MiB) |
| Driver / CUDA del driver | 580.82.07 / 13.0 (`nvidia-smi`) |
| Compilador | `nvcc` release 13.0, V13.0.88 (`nvcc --version`) |
| Bus | PCIe Gen3 x16 (máximo reportado por `nvidia-smi -q`) |
| Motores de copia | `asyncEngineCount = 3` (`cudaGetDeviceProperties`) |
| Sistema operativo | Ubuntu 24.04 (imagen de Colab para GPU) |
| Herramientas | `compute-sanitizer` y Nsight Systems (`nsys`) del toolkit de Colab |

## 2. Compilación y ejecución

Los tres programas compilan sin errores ni warnings con `-O2`. Se agrega `-arch=sm_75` para generar código nativo de la T4 y evitar la compilación JIT del PTX en el primer lanzamiento.

```bash
git clone https://github.com/davidcamiloch/colmenares-post1-u11
cd colmenares-post1-u11
nvcc -O2 -arch=sm_75 -o vectorAdd       src/vectorAdd.cu       && ./vectorAdd
nvcc -O2 -arch=sm_75 -o matMul          src/matMul.cu          && ./matMul
nvcc -O2 -arch=sm_75 -o vectorAddStream src/vectorAddStream.cu && ./vectorAddStream
./vectorAddStream sweep        # barrido con 1, 2, 4, 8, 16 y 32 streams
compute-sanitizer --leak-check full ./vectorAdd        # 0 fugas, 0 errores
compute-sanitizer --tool memcheck ./matMul             # 0 errores
nsys profile -o s4 ./vectorAddStream 4                 # traza de solapamiento
```

En Colab cada comando va en una celda precedido de `!`; el clonado y `%cd colmenares-post1-u11` van en la primera celda.

## 3. Resultados de vectorAdd (Checkpoint 1)

El tiempo de CPU es reloj de pared (`clock_gettime`). El tiempo de kernel es el promedio de 10 lanzamientos medidos con `cudaEvent` después del calentamiento. El tiempo total de GPU incluye las dos copias H2D, el kernel y la copia D2H con memoria de host paginable (`malloc`), como en el código de la guía.

| N | CPU (ms) | GPU kernel (ms) | Ancho de banda del kernel | GPU total con memcpy (ms) | Speedup del kernel | Errores |
|---:|---:|---:|---:|---:|---:|---:|
| 1 048 576 (1M) | 2.06 | 0.051 | 245.3 GB/s | 4.88 | 40× | 0 |
| 4 194 304 (4M) | 8.26 | 0.194 | 259.5 GB/s | 19.20 | 43× | 0 |
| 16 777 216 (16M) | 36.08 | 0.765 | 263.2 GB/s | 73.02 | 47× | 0 |

La línea `N = 1024` de la salida es la ejecución de calentamiento y no entra en la tabla.

## 4. Resultados de matMul (Checkpoint 2)

Cada kernel se mide como el promedio de 20 lanzamientos tras uno de calentamiento. La referencia de CPU usa orden i-k-j y acumula en `double`. Las matrices tienen valores aleatorios en [-1, 1] con semilla fija.

| N | CPU (ms) | Naive (ms) | Tiled TILE=16 (ms) | Speedup tiled/naive | Speedup tiled/CPU | GFLOP/s tiled | Error máx. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 59.53 | 0.825 | 0.532 | 1.55× | 112× | 504.9 | 3.05e-05 |
| 1024 | 869.13 | 8.221 | 4.535 | 1.81× | 192× | 473.5 | 6.48e-05 |

Ambos kernels cumplen el criterio de error menor que 1e-3 frente a la CPU. `compute-sanitizer` reporta 0 errores de acceso en los dos kernels y 0 bytes fugados en `vectorAdd` y `vectorAddStream`.

## 5. Síncrono vs streams (Checkpoint 4)

`vectorAddStream.cu` usa memoria pinned (`cudaHostAlloc`) en las dos versiones; así la comparación aísla el efecto del solapamiento. Cada tiempo es el promedio de 5 repeticiones tras el calentamiento.

| Configuración | Tiempo total (ms) | Reducción frente al síncrono | Errores |
|---|---:|---:|---:|
| Síncrono (H2D + kernel + D2H) | 16.79 | — | 0 |
| 4 streams (configuración de la guía) | 13.45 | 19.9 % | 0 |

| Streams | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Tiempo (ms) | 16.80 | 14.52 | 13.51 | 12.99 | 12.91 | 13.10 |
| Reducción | 0.1 % | 13.6 % | 19.6 % | 22.7 % | 23.2 % | 22.1 % |

Mediciones de apoyo del mismo programa: ancho de banda H2D de 12.34 GB/s con memoria pinned frente a 4.53 GB/s con memoria paginable; volumen transferido de 201 MB (192 MiB); kernel solo de 0.775 ms.

La traza de Nsight Systems de la última iteración con 4 streams muestra el mecanismo. Las copias H2D se ejecutan una tras otra (1.39 a 1.41 ms por chunk de 16.8 MB; unos 12 GB/s). La copia D2H del stream 15 (3.017 a 4.696 ms) corre al mismo tiempo que la copia H2D del stream 16 (2.799 a 5.863 ms). El kernel de cada chunk dura 0.2 ms. La iteración completa termina en 13.42 ms.

## 6. Análisis

El kernel de GPU supera a la CPU por un factor de 40 a 47 en vectorAdd porque la T4 lanza 16 777 216 threads repartidos en 40 SM y su memoria GDDR6 sostiene 263 GB/s medidos. La CPU recorre el arreglo con un solo hilo y queda limitada por su propio ancho de banda de memoria; el factor crece con N porque el costo fijo de lanzamiento (unos microsegundos) pesa menos. En matMul la ventaja llega a 192× porque la operación tiene intensidad aritmética alta: 2N³ operaciones sobre 3N² datos. El tiling divide entre 16 las lecturas de memoria global; la mejora medida sobre el naive es 1.55× en N = 512 y 1.81× en N = 1024. La mejora no llega a 16× porque la caché L1 de Turing ya captura parte de la reutilización del kernel naive.

El tiempo total con `cudaMemcpy` es mayor que el de CPU en todos los tamaños medidos: 73.02 ms frente a 36.08 ms con N = 16M. La suma de vectores lee 8 bytes y escribe 4 bytes por cada suma; con memoria paginable el bus entrega unos 2.8 GB/s efectivos y la transferencia consume el 99 % del tiempo total. Con N pequeño el efecto es igual o peor porque la latencia fija de cada copia no se amortiza. La GPU solo compensa cuando los datos ya residen en el dispositivo o cuando la operación hace mucho cómputo por byte transferido. La regla es mover los datos una vez y reutilizarlos muchas veces en la GPU.

## 7. Decisión técnica 1 — ¿cudaMemcpyAsync con streams o transferencia síncrona para vectorAdd?

El análisis muestra que vectorAdd es memory-bound y que el solapamiento solo oculta la parte menor del pipeline. Con N = 16M se transfieren 201 MB (192 MiB). A 10 a 12 GB/s la transferencia tarda entre 16.8 y 20.1 ms; con los 12.33 GB/s medidos en la T4 tarda 16.3 ms. El kernel tarda 0.775 ms. La transferencia es unas 21 veces mayor que el cómputo.

Si solo se solapara cómputo con copia el ahorro máximo sería el tiempo del kernel: 0.775 ms sobre 16.79 ms; es decir 4.6 %. La reducción medida fue 19.9 % con 4 streams. La diferencia se explica con un dato del entorno: la T4 reporta 3 motores de copia y PCIe es full-duplex. La traza de `nsys` confirma que la D2H de un chunk corre en paralelo con la H2D del siguiente. El ahorro máximo real es entonces el kernel más la copia D2H (0.775 + 5.13 ms). El modelo T = T_H2D + (T_kernel + T_D2H)/S predice 12.36 ms con 4 streams; el valor medido es 13.45 ms.

El estudiante decide no usar streams como optimización principal de vectorAdd. La ganancia existe pero no cambia la conclusión: la GPU total sigue más lenta que dejar los datos en el dispositivo. En matMulTiled el cómputo por chunk crece como O(N³) y puede igualar la transferencia; allí el solapamiento oculta casi todo el tiempo de copia. La regla es usar streams cuando el cómputo por chunk es comparable a su transferencia.

## 8. Decisión técnica 2 — ¿Cuántos streams usar y qué limita la ganancia?

El barrido medido responde la pregunta. De 1 a 2 streams la reducción salta de 0.1 % a 13.6 %; de 2 a 4 sube a 19.6 %; de 4 a 8 a 22.7 %; de 8 a 16 a 23.2 %; con 32 streams baja a 22.1 %. Pasar de 4 a 16 streams no reduce el tiempo de forma proporcional: solo gana 0.6 ms más.

El recurso que limita la ganancia es el bus PCIe Gen3 x16 con sus motores de copia. La T4 tiene 3 motores de copia; uno atiende H2D y otro D2H a la vez. La traza muestra que las copias H2D de distintos streams nunca se solapan entre sí: se ejecutan en fila a unos 12 GB/s. Dividir la transferencia en más streams no crea ancho de banda nuevo; los 134 MB de entrada siempre tardan unos 10.9 ms. Más streams solo reparten esos bytes en trozos más pequeños; el pipeline se llena antes y la cola final es más corta. Con 16 o más streams cada chunk mide 4.2 MB o menos y el costo fijo de lanzar cuatro operaciones por stream empieza a comerse la ganancia; por eso 32 streams es más lento que 16.

El estudiante decide usar 4 streams. Esa cifra captura el 85 % de la mejor reducción medida con un código simple y chunks grandes. La regla es usar tantos streams como motores de copia más uno o dos, no más.

## 9. Desviaciones respecto al código guía

| # | Código guía | Problema detectado | Corrección aplicada |
|---|---|---|---|
| 1 | `vectorAdd.cu` mide la CPU con `clock()` | `clock()` mide tiempo de CPU del proceso, no tiempo de pared; además falta `#include <time.h>` | Se usa `clock_gettime(CLOCK_MONOTONIC)` y se incluyen `time.h` y `math.h` |
| 2 | `vectorAdd.cu` mide un solo lanzamiento del kernel | Con CUDA 12 o superior el primer lanzamiento carga el módulo de forma perezosa e infla el tiempo | Se hace una ejecución de calentamiento y se promedian 10 lanzamientos |
| 3 | `vectorAdd.cu` solo mide el kernel con N = 16M | El README exige N = 1M, 4M y 16M y el tiempo total con `cudaMemcpy` | `run(n)` mide CPU, kernel y total para los tres tamaños |
| 4 | Ningún programa revisa errores de la API | Un fallo de `cudaMalloc` o de lanzamiento pasaría en silencio | Macro `CUDA_CHECK` en cada llamada y `cudaGetLastError()` tras cada lanzamiento |
| 5 | `matMul.cu` es un fragmento sin `main` | No compila solo; no tiene kernel naive ni verificación contra CPU | Programa completo con naive, tiled, referencia en CPU y error máximo |
| 6 | `vectorAddStream.cu` no verifica resultados | El Checkpoint 4 exige resultados correctos en ambas versiones | Comparación contra la referencia de CPU tras cada versión |
| 7 | `vectorAddStream.cu` mide una sola vez y sin calentamiento | La primera versión medida absorbe la carga del módulo y distorsiona la reducción | Calentamiento y promedio de 5 repeticiones |
| 8 | `CHUNK = N / NSTREAMS` | Se pierden elementos si N no es divisible entre el número de streams | El último chunk absorbe el residuo; el número de streams se recibe por argumento |
| 9 | La guía supone PCIe 4.0 x16 | La T4 de Colab es PCIe Gen3 x16 | Se midió el ancho de banda real (12.33 GB/s pinned) y se usó en las decisiones |
| 10 | La guía habla de 192 MB | 3 × 16 777 216 × 4 B = 201.3 MB = 192 MiB | Se reporta en MB y en MiB |
| 11 | La guía estima una reducción con streams cercana al 10 % | La T4 tiene 3 motores de copia y solapa H2D con D2H | Se midió 19.9 % y se explicó con la traza de `nsys` |

## 10. Capturas

| Checkpoint | Capturas |
|---|---|
| 1. Benchmark vectorAdd | `capturas/cp1-entorno-colab.png`, `capturas/cp1-vectoradd.png` |
| 2. matMul con tiling | `capturas/cp2-matmul.png`, `capturas/cp2-sanitizer.png` |
| 3. README y repositorio | `capturas/cp3-estructura-repo.png`, `capturas/cp3-git-log.png` |
| 4. Streams | `capturas/cp4-streams.png`, `capturas/cp4-nsys-traza.png` |
| 5. Decisiones técnicas | `capturas/cp5-barrido-streams.png`, `capturas/cp5-pcie-copy-engines.png` |

## Referencias

- NVIDIA. (2024). *CUDA C++ Programming Guide* (secciones de streams, memoria page-locked y ejecución concurrente). https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA. (s.f.). *NVIDIA T4 Tensor Core GPU*. Recuperado el 19 de noviembre de 2026 de https://www.nvidia.com/en-us/data-center/tesla-t4/
