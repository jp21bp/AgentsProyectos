# Especificaciones del Experimento v1.0

## Indice de Contenidos
* 



## Conceptos Generales
* Recuperacion: el **mecanismo** tecnico para encontrar y seleccionar informacion desde una fuente

* Adquisicion: el **proceso** de decidir y obtener informacion adicional; incluye: recuperacion, cuanta informacion obtener, que tipo de operacion realizar, etc.

* Utilidad Marginal:
$$
\Delta U(q,a) = U(y_{t+1}) - U(y_t)
$$

* Valor Marginal:
$$
MV(q,a) = \frac{\Delta U(q,a)}{C(a)}
$$



## 1. Pregunta de Investigacion
### Pregunta Principal

La investigacion intenta investigar **valor marginal** que se adquiere con informacion adicional:

**Cuales son los momentos en donde adquiriendo informacion adicional resulta en un valor marginal positivo para un agente LLM de pregunta-respuesta, y se podria predecir ese valor antes de pagar el costo de adquisicion?**

Concretamente:

**Dado la evidencia (informacion ya adquerida) y el estado actual del agente, podemos predecir si ejecutando otra operacion para adquerir mas informacion significativamente mejoraria la decision final para justificar el costo de adquisicion?**













## 2. Campo del Primer Experimento
### Incluido
* Pregunta-respuesta textual
* Preguntas multi-saltos y relacionales
* LLM y recuperaciones fija
* Recuperacion densa
* Recuperacion grafica
* Verificacion opcional
* Adquisicion secuencial
* Costo de adquisicion
* Rango de respuesta correcta
* Confianza en respuesta
* Cobertura de evidencia
* Utilidad marginal
* Prediccion de utilidad marginal

### Excluido
* Busquedad autonoma
* Memoria de largo-plazo
* Aprendizaje de trayectorias previas
* Aprendizaje reforzado
* Calibracion
* Multiple agentes
* Framework de planificacion
* Memoria temporal
* Informacion multimodal
* Ambientes complejos de herramientas
* Knowledge graphs dinamicos
* Controladores sofisticados
* Modelos propietarios













## 3. Hipotesis 
### H1: Valor Marginal

H1: Adquiriendo informacion adicional tiene **valor marginal heterogeneo** a traves diferentes preguntas: la adquisicion pueden resultar en utilidad positiva, cero, o negativa a traves diferentes preguntas

Formalmente, dejemos:
* $Q$ = pregunta
* $s_t$ = estado actual del agente
* $a$ = accion de adquisicion
* $y_t$ = respuesta antes de adquisicion
* $y_{t+1}$ = respuesta despues de adquisicion
* $U(y)$ = utilidad de una respuesta
* $C(a)$ = costo de adquisicion

Definimos:
* Utilidad Marginal
$$
\Delta U(Q,a) = U(y_{t+1}) - U(y_t)
$$

* Valor Marginal
$$
MV(Q,a) = \frac{\Delta U(Q,a)}{C(a)}
$$


* Valor Marginal Heterogeneo
$$
MV(Q_i,a) \neq MV(Q_j,a) \quad \text{para preguntas } Q_i, Q_j
$$

<br>

$$
\rightarrow \Delta U(Q_i,a) \neq \Delta U(Q_j,a)
$$


Por ende:
$$
H1: Var(\Delta U(Q,a)) > 0
$$

Donde la varianza se explica parcialmente por propiedades observables dentro el estado actual $s_t$. 



### H2: Predecibilidad

H2: Propiedades observables dentro el estado actual $s_t$ contiene informacion que determina la utilidad de otra aduisicion.

Formalmente, accumulemos las caracteristicas:

$$
X_t =[c_t,r_t,e_t,d_t,o_t,h_t,x_t,q_t]
$$

Donde:

- $c_t$: Confianza del modelo
- $r_t$: Puntaje de recuperacion
- $e_t$: Cobertura de la evidencia
- $d_t$: Estadistica del grafo
- $o_t$: Coincidencia de evidencia
- $h_t$: Complejidad de razonamiento
- $x_t$: Indicadores contradictorios
- $q_t$: Caracteristicas de la pregunta


Por ende:
$$
H2: \exists P(\Delta U > 0 \mid X_t)
$$

que puede predecir significativamente mejor que una base.






