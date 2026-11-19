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

## 5. Cuantización INT8 vs FP32: efecto en throughput y precisión

Las cifras de TOPS dependen de la precisión. En la misma generación de hardware INT8 duplica el pico de FP16 o BF16.

| Hardware | FP32 | FP16/BF16 Tensor | INT8 Tensor | Relación INT8/FP16 | Fuente |
|---|---|---|---|---|---|
| NVIDIA T4 | 8.1 TFLOPS | 65 TFLOPS | 130 TOPS | 2.0× | NVIDIA (s.f.-b) |
| NVIDIA H100 SXM (con sparsity) | 67 TFLOPS | 1 979 TFLOPS | 3 958 TOPS | 2.0× | NVIDIA (s.f.-a) |
| TPU v5e | — | 197 TFLOPS | 393 TOPS | 2.0× | Google Cloud (s.f.-b) |

La exactitud se mide con modelos reales. Wu et al. (2020) cuantizaron ResNet-50 v1.5 y BERT-Large con distintos métodos de calibración post-entrenamiento (PTQ) y con entrenamiento consciente de la cuantización (QAT).

| Modelo | FP32 | INT8 PTQ max | INT8 PTQ entropía | INT8 PTQ percentil 99.99 % | INT8 PTQ percentil 99.999 % | INT8 QAT | Umbral MLPerf |
|---|---|---|---|---|---|---|---|
| ResNet-50 v1.5 (top-1 %) | 76.16 | 75.82 | 76.05 | 75.98 | 75.97 | 76.85 | 99 % de 76.46 = 75.70 |
| BERT-Large (F1) | 91.01 | 85.92 | 37.40 | 89.59 | 90.20 | 90.67 | 99 % de 90.874 = 89.97; 99.9 % = 90.78 |

Fuentes: Wu et al. (2020) para las exactitudes; MLCommons (s.f.) y Han y Kayrish (s.f.) para los umbrales.

En ResNet-50 todas las calibraciones quedan a menos de 0.5 puntos del FP32. En BERT-Large la calibración por entropía hunde el F1 de 91.01 a 37.40. El mismo modelo con percentil 99.999 % pierde solo 0.81 puntos. La cuantización no es un cast: elegir mal la escala de una sola capa destruye el modelo. Un cast directo sin calibración es todavía peor porque ni siquiera ajusta la escala al rango real de las activaciones.

MLPerf exige que un envío cuantizado conserve el 99 % de la exactitud FP32 de referencia; BERT y DLRM tienen además una categoría de 99.9 % (MLCommons, s.f.). La guía afirma que BERT siempre debe cumplir 99.9 %; las reglas publican las dos categorías y NVIDIA reporta ambas (Eassa et al., 2024). La calibración solo puede usar el conjunto de calibración que entrega MLPerf y el método debe publicarse para que sea reproducible (MLCommons, s.f.).

## 6. Decisión técnica — ¿GPU centralizada o FPGA distribuida para el Perfil B?

El análisis concluye que la FPGA distribuida es la opción correcta si el rechazo de piezas tiene implicaciones de seguridad. Las 50 líneas a 10 imágenes por segundo suman 500 imágenes por segundo. Una H200 sostiene 94 620 muestras por segundo de ResNet-50 (Eassa et al., 2024); un modelo tipo ResNet-18 es más liviano. La carga usa menos del 1 % de la capacidad de una GPU Hopper.

El throughput sobra; la latencia no está garantizada. Si la GPU atiende las imágenes una por una con 1.5 ms de servicio, la utilización es 0.75. Un modelo de cola M/D/1 da una espera media de 2.25 ms y un tiempo medio de 3.75 ms; la cola de la distribución supera los 5 ms. Con 1.9 ms de servicio la espera media sube a 18 ms. La GPU solo cumple si ejecuta varias inferencias en paralelo con varios contextos de TensorRT o lotes pequeños.

En capital la GPU gana con claridad. Un servidor con H100 cuesta 30 000 USD instalados (Jarvislabs, 2026). Cincuenta U55C cuestan 400 000 USD con el precio de la guía y 237 350 USD con el precio de lista de AMD (AMD, s.f.). La centralización agrega dos riesgos: un punto único de falla que deja sin inspección a las 50 líneas; jitter del sistema operativo, del driver y de la red entre cámara y servidor.

El estudiante decide FPGA por línea para un sistema de seguridad industrial. Si la inspección es solo control de calidad, la GPU centralizada con un segundo servidor de respaldo es razonable y cuesta menos. La regla es pagar el determinismo solo cuando una latencia fuera de límite cuesta más que el hardware.

## 7. Decisión técnica — ¿Conviene cuantizar a INT8 el modelo del Perfil B?

El análisis recomienda cuantizar a INT8 con una condición: validar el recall de la clase defecto por separado. Si el modelo ya cumple 5 ms en FP16, INT8 no hace falta para cumplir la latencia. Su beneficio es el margen. La T4 pasa de 65 TFLOPS en FP16 a 130 TOPS en INT8 (NVIDIA, s.f.-b); el mismo hardware atiende cerca del doble de imágenes. Ese margen absorbe una ampliación de líneas o un modelo más grande sin comprar tarjetas.

El riesgo está en la métrica. Wu et al. (2020) muestran pérdidas de 0.11 a 0.34 puntos de top-1 en ResNet-50 con PTQ. Ese promedio se calcula sobre todas las clases. La clase defecto es minoritaria; sus activaciones extremas son justo las que la calibración recorta. Una caída de 0.3 puntos en el promedio puede esconder varios puntos de recall en esa clase. El caso de BERT con calibración por entropía prueba que una mala escala puede destruir un modelo aunque otro método funcione.

El estudiante decide exigir tres pruebas antes de pasar a producción: recall y precisión de la clase defecto con su intervalo de confianza; matriz de confusión INT8 contra FP16 sobre el mismo conjunto; comparación de varios métodos de calibración. Si el recall de defecto cae más que el margen acordado con calidad, las capas sensibles se mantienen en FP16. La regla es medir la clase que cuesta dinero, no el promedio.

## Conclusiones

El análisis muestra que la GPU es la opción más versátil y la que lidera MLPerf en throughput; la NPU domina en eficiencia cuando el modelo cabe en su memoria local; la TPU escala mejor dentro de Google Cloud; la FPGA ofrece la única latencia determinística. Las cifras de los fabricantes exigen leer la letra pequeña: sparsity en la H100; precio por chip en la TPU; núcleos frente a TOPS en el Neural Engine.

La Parte 1 confirma en pequeño la misma lección. La T4 multiplica por 47 la CPU en el kernel, pero pierde contra la CPU cuando se cuentan las transferencias. El acelerador correcto es el que reduce el costo total del sistema, no el que publica más TOPS.

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
