# PixelSage  
### Recomendador local y explicable de videojuegos

PixelSage es un **sistema de recomendación de videojuegos que se ejecuta íntegramente en local**, diseñado para ayudar a los usuarios a descubrir juegos acordes a sus gustos utilizando **lenguaje natural**.

PixelSage combina:
- búsqueda semántica mediante embeddings,
- un motor de búsqueda vectorial,
- y un modelo de lenguaje utilizado de forma controlada,

para generar recomendaciones **explicables, transparentes y respetuosas con la privacidad**.

---

## ¿Por qué PixelSage?

Los catálogos actuales de videojuegos contienen **decenas de miles de títulos**, lo que dificulta encontrar juegos adecuados sin invertir una gran cantidad de tiempo.  
Los sistemas de recomendación existentes suelen ser opacos, poco personalizables y dependen de infraestructuras centralizadas.

PixelSage explora una alternativa:

> ¿Es posible construir un recomendador que funcione en local, entienda lenguaje natural y explique por qué recomienda un juego?

---

## ¿Cómo funciona PixelSage?

PixelSage sigue un **pipeline híbrido de recomendación**:

1. **Consulta en lenguaje natural**  
   El usuario describe el tipo de juego que busca.

2. **Modelo de lenguaje como planificador**  
   Un modelo LLM local interpreta la consulta y la traduce a una intención estructurada  
   (qué buscar, cuántos resultados, restricciones).

3. **Búsqueda semántica**  
   Las descripciones de los juegos se representan mediante embeddings y se comparan por similitud.

4. **Filtrado y ajuste por preferencias**  
   Se aplican preferencias explícitas del usuario y restricciones definidas.

5. **Recomendación explicada**  
   El sistema propone uno o dos juegos y explica brevemente por qué encajan con la consulta.

Todo el proceso se ejecuta **en el equipo del usuario**, sin enviar datos a servicios externos.

---

## Principios de diseño

PixelSage se apoya en los siguientes principios:

- **Local-first**  
  El sistema no depende de APIs externas ni de envío de datos.

- **Explicabilidad**  
  Las recomendaciones se basan en pasos claros y comprensibles.

- **Uso controlado de LLMs**  
  El modelo de lenguaje actúa como asistente, no como decisor autónomo.

- **Simplicidad y control**  
  Se prioriza la claridad del sistema frente a soluciones complejas u opacas.

---

## Qué es PixelSage (y qué no)

**PixelSage es:**
- un recomendador basado en contenido,
- orientado a búsquedas semánticas,
- diseñado para uso local y demostraciones académicas.

**PixelSage no es:**
- un sistema de filtrado colaborativo,
- una plataforma comercial,
- un servicio en la nube.

---

## Ejecución del proyecto

PixelSage se ejecuta en local utilizando Python y Streamlit.

Pasos básicos:

```bash
pip install -r requirements.txt
python -m src.ingest.generate_embeddings
streamlit run streamlit_app.py
