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
* Recuperacion de verificacion (Opcional)
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
* $q$ = pregunta
* $s_t$ = estado actual del agente
* $a$ = accion de adquisicion
* $y_t$ = respuesta antes de adquisicion
* $y_{t+1}$ = respuesta despues de adquisicion
* $U(y)$ = utilidad de una respuesta
* $C(a)$ = costo de adquisicion

En este experimento existen 3 acciones de adquisiciones $a$:
* Recuperacion densa
* Recuperacion grafica
* Recuperacion de verificacion (Opcional)

Definimos:
* Utilidad Marginal
$$
\Delta U(q,a) = U(y_{t+1}) - U(y_t)
$$

* Valor Marginal
$$
MV(q,a) = \frac{\Delta U(q,a)}{C(a)}
$$


* Valor Marginal Heterogeneo
$$
MV(q_i,a) \neq MV(q_j,a) \quad \text{para preguntas } q_i, q_j
$$

<br>

$$
\rightarrow \Delta U(q_i,a) \neq \Delta U(q_j,a)
$$


Por ende:
$$
H1: Var(\Delta U(q,a)) > 0
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















## 4. Metricas
### Utilidad

Metrica principal: dado una respuesta $y$ para una pregunta $Q$, la utilidad se puede medir en forma binaria dependiendo en que tan correcto este la respuesta:

$$
U(y) = 
\begin{cases}
1, \text{si la respuesta esta correcta} \\
0, \text{si la respuesta no esta correcta}
\end{cases}
$$

Metricas secundarias:
* Respuesta exacta
* F1
* Correcion de evidencia/apoyo
* Calibracion de confianza


Si  $U_{pos}$ y $U_{pre}$ son las utilidades despues y antes de hacer una aduiqisicion, respetivamente, entonces:
$$
\Delta U = U_{pos} - U_{pre}
$$

Con la metrica binaria:

| Antes | Despues | ΔU |
| --- | --- | --- |
| incorrecta | incorrecta | 0 |
| incorrecta | correcta | +1 |
| correcta | correcta | 0 |
| correcta | incorrecta | -1 |



### Costo

El costo es individual a cada componente del experimento:
* Recuperacion densa
    - Numero de documentos recuperados
    - Numero de documentos distintos
    - Tokens utilizados
* Recuperacion grafica
    - Vertices expandidas
    - Aristas travesadas
    - Entidades distintas
    - Pasos en la recuperacion
    - Tokens utilizados
* Verificacion
    - Numero de llamadas
    - Documentos adicionales
    - Tokens adicionales
* Cost del Modelo
    - Numero de llamadas al LLM
    - Tokens de ingreso
    - Tokens de egreso
    - Tokens totales
    - Latencia

Estos costos individuales se utilizan para crear funciones de costo, por ejemplo:

* 
$$
C_{\text{tokens}}
=
\text{tokens ingreso}+\text{tokens egreso}
$$

* 
$$
C_{\text{llamadas}}
=
\#\text{llamadas LLM}
$$


*  
$$
C_{\text{recuperacion}}
=
\#\text{documentos recuperados}
$$


















## 5. Datos
### Conjunto 'HotpotQA'
Se utilizara 'HotpotQA' por sus siguiente propiedades:
* Preguntas multi-salto
* Documentos apoyantes
* Documentos distratores
* Multiple estructuras de razonamiento
* Evaluacion QA establecida
* Magnitud manegable
* Literatura existente

Separacion de datos:
* Desarrollo: 500 preguntas
* Piloto: 100 preguntas
* Experimento final: 1_000 - 2_000 preguntas, dependiendo en recursos computacionales


### Seleccion de preguntas
Las preguntas se van a seleccionary dependiendo en la complejidad de su informacion:
* 1-salto/baja complejidad
* 2-saltos
* multi-saltos
* razonamiento entre entidades
* preguntas de comparacion
* preguntas con distractores

Si la metadata del conjunto no hace distincciones claras entre diferentes complejidades, entonces las categorias se tienen que modificar.


Proposito de esta seleccion:
**El valor marginal de adquisicion depende en la complejidad de la pregunta?**





















## 6. Modelo

El modelo seleccionada tendra las siguiente caracteristica:
* Pequeno
* Open source
* Pulido con instrucciones
* Ejecucion local
* Especializado para QA
* Inferencia deterministica
* Tokenizer disponible
* Implementable con HuggingFace
* Sin entrenar, esta fijo

Condiciones adicionales:
* Temperature = 0
* Top_p = 1
* Tamano maximo de egreso fijo
* Se utiliza el mismo prompt dentro cada criterio
* Lo unico que cambia entre criterios es la informacion que se provee a los modelos
    - Esto es esencial para interpretacion causal
























## 7. Condiciones Experimentales

El experimento cosiste de estados de informacion secuencial

### Condicion 0 - sin recuperacion

El modelo respode a la pregunta $q$ utilizando solo la informacion dentro sus pesos parametricos.

Registrar los siguientes aspectos:
* respuesta
* confianza
* tokens
* latencia
* nivel de correcto

Este estado se asigna $S_0$. 


### Condicion 1 - recuperacion densa

Recuperar los top-$k$ documentos, donde $k=3$.

Para cada pregunta $q$:
$$
q \rightarrow Recuperador(q) \rightarrow E_{D_k} \rightarrow LLM(q,E_{D_k})
$$

donde $E_{D_k}$ es la evidencia de la recuperacion densa de top-$k$ documentos.

Registrar los siguiente aspectos:
* Documentos recuperados
* Puntaje de recuperacion
* Classificacion numeral
* Tokens del documento
* Respuesta
* Confianza
* Nivel de correcto
* Latencia
* Tokens LLM

Este estado se asigna $S_1$. 


### Condicion 2 - Recuperacion grafica

Crear un grafo del mismo conjunto 'HotpotQA'. El objetivo **no es** comparar el tipo de recuperacion, el objetivo es investigando si existen caracteristicas observables dentro cada estado que pueden predecir el valor marginal de cada adquisicion. 


El grafo $G=(V,E)$ se creara de las entidades $V$ y las relaciones $E$ entre entidades.

Para cada pregunt $q$, se:
1. Identificara las entidades
2. Ubicara los nodos(entidades) correspondientes
3. Expandira una vecindad acotada
4. Recuperara evidencia associada
5. Proveera esa evidencia al mismo LLM

En otras palabras:
$$
q \rightarrow Recuperador(q) \rightarrow E \rightarrow LLM(q,E)
$$

donde $E_G$ es la evidencia de la recuperacion grafica. 



Registrar los siguiente aspectos:
* Semilla aleatoria
* Nodos expandidos
* Aristas travesadas
* Salto maximo
* Evidencia recuperada
* Tokens de la evidencia
* Respuesta
* Confianza
* Nivel de correcto
* Latencia

Este estado se asigna $S_2$. 


### Condicion 3 - Verificacion (Opcional)

Dentro cada estado de evidencia $\{S_1, S_2\}$ actual, se hace una operacion adicional de verificacion. 

Prompt simple: "Recuperar evidencia adicional con la intenciones de verificar la respuesta actual"

Por cada estado $S_{1,2}$:
$$
S_{1,2} \rightarrow \text{recuperacion de verificacion} \rightarrow S_3
$$


Registrar los siguiente aspectos:
* Prompt/query de verificacion
* Evidencia recuperada
* Evidence duplicada
* Indicadores de contradiccion
* Tokens adicionales
* LLamadas de recuperacion adicionales
* Respuesta final
* Nivel de correcto


















## 8. Evaluacion Experimental
### Datos Coleccionados
EL dato clave no es la respuesta final, es la **transicion** entre diferentes estados de evidencia $S$ al seleccionar una operacion de adquision/recuperacion $a$.

$$
S_t \rightarrow a \rightarrow S_{t+1}
$$

Para **cada** transicion, registramos:

$$
(q,S_t,a,S_{t+1},\Delta U(q,a),C)
$$

Esto crea un conjunto de datos que corresponde a **decisiones de adquisicion informatico**.

### Clasificando datos

Utilizando la metrica binaria de utilidad:
* $\Delta U \in \{-1,0,+1\}$
* Adquisicion que ayuda: $\Delta U=+1$
* Adquisicion quneutral: $\Delta U=0$
* Adquisicion que no ayuda: $\Delta U=-1$

### Primeros resultados

Dentro de cada categoria:
* Complejidad de pregunta
* Confianza de recuperacion
* Grado de grafo
* Cobertura de evidencia
* Confianza de respuesta

Se grafica:
$$
P(\Delta U=+1) = \frac{len(U=+1)}{len(U)}
$$



Objetivo: Se quiere identificar si **algunos estados** beneficiaron de adquisicion adicional
* Resultado negativo: adquisicion adicional no le ayuda a ningun estado
* Resultado debil: adquisicion adicional les ayuda a todos los estados
* Resultado fuerte: existen algunos estados que se benefician de adquisicon adicional, mientras otros estados no
    - En este caso, el siguiente paso es investigar las propiedades de los estados que puedan predecir un beneficio antes de ejecutar la adquisicion.





