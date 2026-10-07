# Predicción de Churn - MLOps Baseline

Este repositorio contiene la migración a un script ejecutable (`train.py`) de un modelo Random Forest para predecir el abandono de clientes (Churn), instrumentado con `MLFlow` para registrar parámetros, métricas y artefactos.

## Reproducibilidad y Requisitos
Todas las dependencias están fijadas para garantizar reproducibilidad.
1. Crea un entorno virtual: `python3.12 -m venv TrainBaseline`
2. Activa el entorno: `source TrainBaseline/bin/activate`
3. Actualizar los paquetes: `python -m pip install --upgrade pip setuptools wheel`
4. Instala las dependencias: `python -m pip install -r requirements.txt`

## Ejecución de las Corridas (Experimentación)
Ejecuta secuencialmente los siguientes comandos en tu terminal. El script dividirá el dataset automáticamente en Train (67%), Validation (22%) y Test (10%).

**1. Baseline (Valores por defecto):**
```bash
python train.py --n-estimators 100 --max-depth 10 --min-samples-split 2
```

**2. Subajuste (Modelo muy simple):**
```bash
python train.py --n-estimators 50 --max-depth 5 --min-samples-split 2
```

**3. Modelo Balanceado (Mayor profundidad, más árboles):**
```bash
python train.py --n-estimators 200 --max-depth 8 --min-samples-split 5
```

**4. Riesgo de Sobreajuste (Profundidad muy alta):**
```bash
python train.py --n-estimators 150 --max-depth 20 --min-samples-split 2
```

**5. Modelo Conservador Regularizado:**
```bash
python train.py --n-estimators 100 --max-depth 12 --min-samples-split 10
```

## Conclusión del Experimento
Tras evaluar las 5 corridas en la interfaz de MLFlow, el modelo que presentó el mejor desempeño en el conjunto de prueba fue el **Baseline (Corrida 1)** (`max_depth=10`). Se comprobó que una profundidad menor a 8 genera un subajuste severo en la clase minoritaria (caída de Recall), mientras que una profundidad de 20 sobreajusta el modelo, deteriorando las métricas de Validación.