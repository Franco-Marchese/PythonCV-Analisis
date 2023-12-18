import cv2
import numpy as np
import pandas as pd
import requests

# Cargar el clasificador de automóviles preentrenado
cascade_automovil = cv2.CascadeClassifier('haarcascade_car.xml')

# Función para obtener el color dominante del frame
def obtener_color_dominante(frame):
    pixeles = frame.reshape((-1, 3))
    unicos, cuentas = np.unique(pixeles, axis=0, return_counts=True)
    color_dominante = unicos[np.argmax(cuentas)]
    return tuple(color_dominante)

# Función para obtener el nombre del color basado en el código RGB utilizando una API
def obtener_nombre_color(rgb):
    url = f'https://www.thecolorapi.com/id?rgb={rgb[0]},{rgb[1]},{rgb[2]}'
    
    try:
        respuesta = requests.get(url)
        datos_color = respuesta.json()
        
        if 'name' in datos_color:
            return datos_color['name']['value']
    except Exception as e:
        print(f"Error al obtener nombre del color: {e}")
    
    # En caso de error o si no se puede obtener el nombre, devolver un valor predeterminado
    return 'Desconocido'

# Inicializar el objeto para realizar la captura de video
cap = cv2.VideoCapture('your/video.mp4')

# Inicializar el objeto para el algoritmo k-means
criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# Número de clústeres para k-means
k = 3

# Inicializar variables para el seguimiento de autos
contador_auto = 0
datos_autos = []

while True:
    # Leer un frame del video
    ret, frame = cap.read()

    # Verificar si el frame se ha leído correctamente
    if not ret:
        break

    # Convertir a espacio de color HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detectar automóviles en el frame
    autos = cascade_automovil.detectMultiScale(hsv_frame, 1.1, 3)

    # Dibujar rectángulos alrededor de los automóviles y asignarles el color dominante
    for (x, y, w, h) in autos:
        id_auto = contador_auto
        contador_auto += 1

        # Obtener la región de interés (ROI) del automóvil
        roi = frame[y:y + h, x:x + w]

        # Obtener el color dominante de la región de interés
        color_dominante = obtener_color_dominante(roi)
        datos_autos.append({'AutoID': id_auto, 'Color': color_dominante})

        # Convertir el color dominante a una tupla de enteros
        color_dominante_int = tuple(map(int, color_dominante))

        # Dibujar rectángulos alrededor de los automóviles y asignarles el color dominante
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_dominante_int, 2)
        # Dibujar texto personalizado en los rectángulos
        cv2.putText(frame, f'Auto {id_auto}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_dominante_int, 2)

    # Mostrar el frame con los rectángulos dibujados
    cv2.imshow('Detección de Autos', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

# Crear DataFrame de Pandas con los datos de los autos
df_autos = pd.DataFrame(datos_autos)

# Agregar una nueva columna con el nombre del color
df_autos['NombreColor'] = df_autos['Color'].apply(obtener_nombre_color)

# Imprimir Data Frame generado para el análisis de conteo por color
print("\nData Frame para análisis de conteo")
print(df_autos)

# Imprimir conteo por color
print("\nResultados por color:")
conteo_colores = df_autos["NombreColor"].value_counts()
print(conteo_colores)
