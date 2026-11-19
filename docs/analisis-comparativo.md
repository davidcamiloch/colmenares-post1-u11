# Análisis comparativo de aceleradores de hardware para inteligencia artificial: GPU, TPU, NPU y FPGA

David Camilo Colmenares Hernández · Arquitectura de Computadores 1154507-B · Unidad 11 · Universidad Francisco de Paula Santander

## Introducción

El documento compara cuatro familias de aceleradores para cargas de inteligencia artificial: la GPU de propósito general con Tensor Cores; la TPU basada en arreglos sistólicos; la NPU integrada en un SoC; la FPGA reconfigurable. El análisis usa especificaciones oficiales de los fabricantes, resultados publicados de MLPerf Inference v4.1 y precios consultados en noviembre de 2026. Las cifras de la propia T4 provienen de las mediciones de la Parte 1 de este repositorio.

La tesis del análisis es simple. Ningún acelerador gana en todas las dimensiones. La GPU lidera en throughput bruto y en ecosistema; la TPU en escala y eficiencia dentro de la nube de Google; la NPU en consumo por inferencia en el dispositivo; la FPGA en latencia determinística. La elección correcta depende del perfil de carga y del costo de equivocarse.

## 1. Caracterización de cada acelerador

La GPU ejecuta miles de hilos en modelo SIMT y agrega Tensor Cores que resuelven multiplicaciones de matrices pequeñas en un ciclo. La TPU organiza sus unidades MXU como arreglos sistólicos: los datos fluyen entre celdas vecinas y cada peso se reutiliza sin volver a la memoria. La NPU fija en silicio un flujo de datos para convolución y atención; renuncia a la generalidad a cambio de pocos vatios. La FPGA no trae una arquitectura de cómputo fija; el diseñador sintetiza el pipeline exacto del modelo con LUT, DSP y BRAM.

| Acelerador | Paradigma | Unidad de cómputo | Precisión y rendimiento pico | Memoria | Potencia | Modelo de programación |
|---|---|---|---|---|---|---|
| GPU NVIDIA H100 SXM | SIMT masivo | CUDA Cores + Tensor Cores de 4.ª generación | BF16 1 979 TFLOPS; INT8 3 958 TOPS (ambas con sparsity) | 80 GB HBM3; 3.35 TB/s | Hasta 700 W | CUDA; cuDNN; TensorRT |
| GPU NVIDIA T4 (usada en la Parte 1) | SIMT | 2 560 CUDA Cores + 320 Tensor Cores | FP32 8.1 TFLOPS; FP16 65 TFLOPS; INT8 130 TOPS | 16 GB GDDR6; 320 GB/s | 70 W | CUDA; TensorRT |
| TPU v5e (Google) | Arreglo sistólico | 1 TensorCore con 4 MXU | BF16 197 TFLOPS; INT8 393 TOPS | 16 GiB HBM; 819 GB/s | No publicada por chip | XLA; JAX; TensorFlow; PyTorch/XLA |
| NPU Apple M3 / M3 Ultra | Dataflow fijo | Neural Engine de 16 núcleos (M3) y 32 núcleos (M3 Ultra) | 18 TOPS (M3); 36 TOPS (M3 Ultra) | Memoria unificada; 819.3 GB/s en M3 Ultra | No publicada para el Neural Engine | Core ML |
| FPGA AMD Alveo U55C | Reconfigurable | 1 304 K LUT; 9 024 DSP | Depende del diseño sintetizado | 16 GB HBM2; 460 GB/s | 150 W máx.; 115 W típica | Vitis HLS; Verilog/VHDL; Vitis AI |

Fuentes de la tabla: NVIDIA (s.f.-a) para la H100; NVIDIA (s.f.-b) para la T4; Google Cloud (s.f.-b) para la TPU v5e; ("Apple M3", s.f.) y Haslam (2025) para el Neural Engine; AMD (s.f.) para la U55C.

El análisis encuentra dos diferencias con la tabla de la guía. El Neural Engine del M3 tiene 16 núcleos y no 18; la cifra 18 corresponde a los TOPS ("Apple M3", s.f.). Las cifras de Tensor Core que NVIDIA publica para la H100 asumen sparsity estructurada; el valor denso es la mitad (NVIDIA, s.f.-a). Comparar TOPS de distintos fabricantes sin aclarar precisión y sparsity produce conclusiones falsas.

La jerarquía de memoria explica gran parte del rendimiento real. La H100 entrega 3.35 TB/s y la TPU v5e 819 GB/s por chip; la U55C 460 GB/s; la T4 320 GB/s. La Parte 1 midió 263 GB/s en el kernel vectorAdd de la T4; es el 82 % de su pico. Un modelo con poca reutilización de datos queda acotado por esa cifra y no por los TOPS.

## 2. Benchmarks MLPerf Inference v4.1

MLPerf Inference mide sistemas completos con cuatro escenarios: single-stream; multistream; server; offline (Reddi et al., 2020). El escenario offline mide throughput sin restricción de latencia. El escenario server envía consultas con llegada aleatoria y mide el throughput que el sistema sostiene sin violar el límite de latencia de cola: 15 ms para ResNet-50 y 130 ms para BERT (Han & Kayrish, s.f.). El escenario single-stream reporta el percentil 90 de latencia (MLCommons, s.f.).

| Modelo (v4.1, Closed, Datacenter) | Sistema | Server | Offline | Por acelerador (offline) | Fuente |
|---|---|---|---|---|---|
| ResNet-50 v1.5 | 8 × H200 | 632 229 consultas/s | 756 960 muestras/s | 94 620 muestras/s | Eassa et al. (2024) |
| ResNet-50 v1.5 | 1 × speedAI240 Preview (Untether AI) | — | 70 348 muestras/s | 70 348 muestras/s | Genkina (2024); Byrne (2024) |
| BERT-Large 99 % | 8 × H200 | 57 609 consultas/s | 73 310 muestras/s | 9 164 muestras/s | Eassa et al. (2024) |
| BERT-Large 99.9 % | 8 × H200 | 51 212 consultas/s | 63 950 muestras/s | 7 994 muestras/s | Eassa et al. (2024) |
| Stable Diffusion XL | 8 × H200 | 16.78 consultas/s | 17.42 muestras/s | 2.18 muestras/s | Eassa et al. (2024) |
| Stable Diffusion XL | TPU v6e Trillium (preview) | La H200 es cerca de 1.9 veces más rápida por acelerador | — | — | Byrne (2024) |

| Eficiencia energética (ResNet-50, categoría Power) | Aceleradores | Consultas/s | Potencia del sistema | Consultas/s por vatio |
|---|---|---|---|---|
| NVIDIA H200-SXM-141GB | 8 | 480 131 | 5 013.79 W | 95.76 |
| Untether AI speedAI240 Slim | 6 | 309 752 | 985.52 W | 314.30 |

Fuente de la tabla de eficiencia: Genkina (2024).

En ResNet-50 la GPU H200 lidera en throughput por sistema y por acelerador. La NPU de Untether AI lidera en eficiencia con 3.28 veces más consultas por vatio. La razón es arquitectónica: su diseño at-memory pone los procesadores al lado de la SRAM y elimina el movimiento de datos que domina el consumo de una GPU (Genkina, 2024). ResNet-50 cabe en esa memoria local; por eso la ventaja aparece en ese modelo.

En BERT-Large la GPU lidera porque la atención exige matrices grandes y mucha memoria de alto ancho de banda. La versión 99.9 % pierde 13 % de throughput frente a la de 99 % en la misma H200; la exigencia de exactitud obliga a usar más precisión en las capas sensibles. Ningún otro acelerador del análisis publicó BERT en esta ronda; la eficiencia en este modelo no tiene comparación directa.

En Stable Diffusion XL la H200 rinde cerca de 1.9 veces lo que rinde una Trillium por acelerador (Byrne, 2024). La guía pide latencias p99 de BERT; MLPerf no publica la latencia p99 de cada sistema en el escenario server. Publica el throughput sostenido bajo el límite de 130 ms. El documento reporta ese dato real en lugar de estimar una latencia.

## 3. Análisis costo-rendimiento y TCO

El rendimiento en TOPS no basta para decidir. El análisis combina tres costos: adquisición; energía; desarrollo y mantenimiento. Los precios se consultaron en noviembre de 2026.

| Acelerador | Costo de adquisición o alquiler | Fuente | Potencia |
|---|---|---|---|
| NVIDIA H100 SXM | 30 000 USD instalados por GPU (supuesto de presupuesto); alquiler desde 3.49 USD por GPU-hora | Jarvislabs (2026) | 700 W |
| TPU v5e | 1.20 USD por chip-hora bajo demanda; 0.54 USD con compromiso de 3 años | Google Cloud (s.f.-a) | Incluida en la tarifa |
| TPU v4 pod | 3.22 USD por chip-hora bajo demanda | Google Cloud (s.f.-a) | Incluida en la tarifa |
| FPGA AMD Alveo U55C | 4 747 USD | AMD (s.f.) | 115 W típica |
| Apple Mac Studio M3 Ultra | 3 999 USD con 96 GB de memoria unificada | Haslam (2025) | No publicada para el Neural Engine |

La guía estima la U55C en 8 000 USD y la TPU v4 en 32 USD por hora. El precio de lista de AMD es 4 747 USD. La tarifa de Google es por chip; 32 USD por hora corresponde a un slice de unos 10 chips v4.

El costo de energía a 3 años se calcula con 26 280 horas; una tarifa supuesta de 0.15 USD por kWh; un PUE supuesto de 1.5 para el centro de datos. Ambos supuestos son explícitos y el lector puede cambiarlos.

$$E_{3a} = P \times 26\,280\ \text{h} \times 0.15\ \tfrac{\text{USD}}{\text{kWh}} \times 1.5$$

Con esa fórmula una H100 a 700 W consume 4 139 USD en 3 años; una U55C a 115 W consume 680 USD; una T4 a 70 W consume 414 USD.

### Escenario (a): startup de IA con inferencia en la nube a menos de 100 solicitudes por segundo

Una H200 sostiene 94 620 muestras por segundo de ResNet-50 (Eassa et al., 2024). Cien solicitudes por segundo usan menos del 0.2 % de esa capacidad. Comprar una H100 cuesta 34 139 USD a 3 años con energía y deja la GPU ociosa casi todo el tiempo. Un chip TPU v5e reservado 3 años cuesta 14 191 USD y no exige operar hardware. Alquilar una H100 todo el tiempo costaría 91 717 USD.

El análisis concluye que la startup debe consumir aceleradores pequeños en la nube y pagar por uso: un chip TPU v5e o una instancia con T4. El costo dominante a esta escala es el de ingeniería; la nube lo reduce porque elimina compras, refrigeración y repuestos.

### Escenario (b): empresa con inferencia en el borde

En un smartphone o una cámara la NPU ya viene dentro del SoC. Su costo marginal es casi cero y su consumo está en pocos vatios. Un Neural Engine de 18 TOPS basta para clasificación y detección ligeras. En una cámara industrial con requisito de latencia fija la FPGA ofrece 4 747 USD más 680 USD de energía por nodo a 3 años. En un vehículo autónomo la carga mezcla varias redes y sensores; ahí una GPU embebida o un SoC con NPU grande da más flexibilidad.

La regla del escenario de borde es usar la NPU del SoC cuando existe; la FPGA cuando la latencia debe ser determinística; la GPU cuando la carga cambia a menudo.

## 4. Recomendación fundamentada por perfil

### Perfil A: laboratorio universitario con 10 000 USD, visión y NLP hasta 7B parámetros

Recomendación: GPU NVIDIA H100 alquilada por horas, complementada con una estación local modesta.

1. Memoria. Un modelo de 7B en BF16 ocupa 14 GB de pesos. El ajuste fino completo con Adam requiere cerca de 16 bytes por parámetro: 112 GB. La H100 con 80 GB admite inferencia y ajuste fino con LoRA; la TPU v5e con 16 GiB por chip obliga a repartir el modelo entre varios chips.
2. Costo. Una H100 no cabe en el presupuesto: 30 000 USD instalados. Los 10 000 USD compran 2 865 horas de H100 a 3.49 USD por hora (Jarvislabs, 2026); suficientes para un año de experimentos de un grupo pequeño.
3. Ecosistema. La mayoría del código de investigación en visión y NLP se publica para PyTorch con CUDA. El tiempo del investigador cuesta más que la hora de GPU.

Limitaciones: las horas se agotan; los datos sensibles salen del campus; la disponibilidad de GPU varía. Un Mac Studio M3 Ultra de 3 999 USD con 96 GB ejecuta inferencia de modelos 7B localmente, pero su Neural Engine no sirve para entrenar y el backend de PyTorch para Apple es más lento que CUDA.

### Perfil B: manufactura con inspección visual a menos de 5 ms en 50 líneas

Recomendación: FPGA distribuida, una por línea, con modelo cuantizado a INT8 validado por clase.

1. Latencia determinística. La FPGA ejecuta el pipeline fijo en hardware sin sistema operativo ni planificador. La latencia máxima es igual a la típica.
2. Aislamiento de fallas. Una tarjeta dañada detiene una línea, no cincuenta.
3. Consumo y ubicación. Una U55C consume 115 W típicos y no necesita red hasta un servidor central; la imagen se procesa junto a la cámara.

Limitaciones: el desarrollo en HLS es más lento que en CUDA; cambiar de modelo exige volver a sintetizar; el capital inicial es mayor. Las decisiones técnicas de las secciones 6 y 7 desarrollan este perfil.

### Perfil C: proveedor de nube que entrena modelos de 70B en clústeres de 512 aceleradores

Recomendación: GPU H100 o H200 con NVLink y red InfiniBand; la TPU es la alternativa solo dentro de Google Cloud.

1. Comunicación colectiva. La H100 ofrece 900 GB/s de NVLink por GPU (NVIDIA, s.f.-a). El entrenamiento de 70B parte el modelo entre GPU y sincroniza gradientes en cada paso; el ancho de banda entre aceleradores define la escalabilidad.
2. Memoria. 80 GB de HBM3 a 3.35 TB/s por GPU permiten paralelismo de tensor con particiones grandes.
3. Mercado. Un proveedor de nube vende capacidad a clientes que traen código CUDA; la GPU maximiza la demanda.

Limitaciones: 512 H100 suman 15.36 millones de USD de capital y 358 kW solo en GPU. La TPU v4 escala a 4 096 chips con conmutadores ópticos y usa entre 1.3 y 1.9 veces menos energía que la A100 en sistemas de tamaño similar (Jouppi et al., 2023); sin embargo solo se alquila en Google Cloud y no se puede comprar.

## Referencias

AMD. (s.f.). *AMD Alveo U55C high performance compute card*. Recuperado el 19 de noviembre de 2026 de https://www.amd.com/en/products/accelerators/alveo/u55c/a-u55c-p00g-pq-g.html

Apple M3. (s.f.). En *Wikipedia*. Recuperado el 19 de noviembre de 2026 de https://en.wikipedia.org/wiki/Apple_M3

Byrne, J. (2024, 20 de septiembre). *MLPerf reveals Blackwell gains, Untether's power efficiency*. XPU.pub. https://xpu.pub/2024/09/20/mlperf-4-1/

Eassa, A., Nanjappa, A., Jiang, Z., Zhang, Y., Yang, J., Kong, Z., & Xu, S. (2024, 28 de agosto). *NVIDIA Blackwell platform sets new LLM inference records in MLPerf Inference v4.1*. NVIDIA Technical Blog. https://developer.nvidia.com/blog/nvidia-blackwell-platform-sets-new-llm-inference-records-in-mlperf-inference-v4-1/

Genkina, D. (2024, 28 de agosto). *AI inference competition heats up*. IEEE Spectrum. https://spectrum.ieee.org/new-inference-chips

Google Cloud. (s.f.-a). *Cloud TPU pricing*. Recuperado el 19 de noviembre de 2026 de https://cloud.google.com/tpu/pricing

Google Cloud. (s.f.-b). *TPU v5e*. Recuperado el 19 de noviembre de 2026 de https://docs.cloud.google.com/tpu/docs/v5e

Han, F., & Kayrish, M. (s.f.). *Dell EMC PowerEdge R750xa server for inference on AI applications*. Dell Technologies InfoHub. Recuperado el 19 de noviembre de 2026 de https://www.dell.com/en-us/infohub/articles/dell-emc-poweredge-r750xa-server-for-inference-on-ai-applications

Haslam, K. (2025, 5 de marzo). *Mac Studio M4 (and M3): What you need to know*. Macworld. https://www.macworld.com/article/2204696/m4-m3-mac-studio-design-processor-specs-price.html

Jarvislabs. (2026). *NVIDIA H100 price guide 2026: GPU costs, cloud pricing and buy vs rent*. https://jarvislabs.ai/blog/h100-price

Jouppi, N. P., Kurian, G., Li, S., Ma, P., Nagarajan, R., Nai, L., Patil, N., Subramanian, S., Swing, A., Towles, B., Young, C., Zhou, X., Zhou, Z., & Patterson, D. (2023). *TPU v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings*. arXiv. https://arxiv.org/abs/2304.01433

MLCommons. (s.f.). *MLPerf inference rules*. GitHub. Recuperado el 19 de noviembre de 2026 de https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc

NVIDIA. (s.f.-a). *NVIDIA H100 Tensor Core GPU*. Recuperado el 19 de noviembre de 2026 de https://www.nvidia.com/en-us/data-center/h100/

NVIDIA. (s.f.-b). *NVIDIA T4 Tensor Core GPU*. Recuperado el 19 de noviembre de 2026 de https://www.nvidia.com/en-us/data-center/tesla-t4/

Reddi, V. J., Cheng, C., Kanter, D., Mattson, P., Schmuelling, G., et al. (2020). *MLPerf inference benchmark*. arXiv. https://arxiv.org/abs/1911.02549

Wu, H., Judd, P., Zhang, X., Isaev, M., & Micikevicius, P. (2020). *Integer quantization for deep learning inference: Principles and empirical evaluation*. arXiv. https://arxiv.org/abs/2004.09602
