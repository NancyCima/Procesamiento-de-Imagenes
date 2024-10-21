import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

"""Se tiene una serie de exámenes resueltos, en formato de imagen, y se
pretende corregirlos de forma automática por medio de un script en python.
El examen múltiple choice consta de 10 preguntas con cuatro opciones para
cada una de ellas (A, B, C, D). La plantilla también tiene un encabezado con
tres campos para completar datos personales (Name, Date y Class).

Para esto, asuma que las respuestas correctas son las siguientes:
1.  C
2.  B 
3.  A 
4.  D 
5.  B 
6.  B 
7.  A 
8.  B 
9.  D 
10. D

Observacion:
En el caso que alguna respuesta tenga marcada más de una opción, la misma se considera
como incorrecta, de igual manera si no hay ninguna opción marcada.
"""


"""
Datos: Tenemos un total de 5 imagenes de examenes.

Analisis:
Las partes relevantes de cada examen son:
    #La respuesta a cada una de las 10 pregunta
    #Los 3 campos del encabezado

Las mismas se analizaran con herramientas de procesamiento de imagenes para
obtener los resultados.

Resultados:
    a) Correcion de cada pregunta
    b) Correccion de los 3 encabezados
    c) Obtener a) y b) para todos los examenes
    d) Crear imagen informando la aprobacion o desaprobacion de los alumnos
"""


def obtener_celdas(img, ver_bin = False, ver_celdas = False):
    """A partir de la imagen de un examen, almacena el numero de pregunta, crop
    y las coordenadas de cada celda en una lista y la devuelve como resultado.
    Como agregado, se puede especificar si se desea ver la imagen binaria
    y las imagenes de cada celda
    """
    #Umbralamos la imagen
    img_bin = img < 100 #Elegimos 100 porque obtenemos las lineas de 1 pixel de espesor

    #Mostramos los cambio
    if ver_bin:
        plt.figure()
        ax = plt.subplot(121)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Imagen original')
        plt.subplot(122, sharex=ax, sharey=ax)
        plt.imshow(img_bin, cmap='gray'), plt.title('Imagen + umbralado < 100')
        plt.savefig('Imagen+umbralado100.png')
        plt.show()

    #Sumamos la cantida de pixeles de cada colmna y fila respectivamente
    sum_cols = np.sum(img_bin,0) 
    sum_rows = np.sum(img_bin,1)

    #Umbralamos el resultado para obtener los indices adecuados
    sum_cols_bin = sum_cols > 500 #Lineas de mas de 500 pixeles
    sum_rows_bin = sum_rows > 230 #Lineas de mas de 230 pixeles

    #Obtenemos los indices de las lineas verticales u horizontales respectivamente
    i_cols = np.where(sum_cols_bin)[0]
    i_rows = np.where(sum_rows_bin)[0]

    #Elimino el primer indice correspondiente al encabezado y los indices con valores consecutivos
    i_rows = i_rows[1:]
    i_rows = np.delete(i_rows, np.argwhere(np.ediff1d(i_rows) <= 1) + 1)

    """Lista en donde almaceno una estructura de la siguiente forma: (id,img,coord)
    Donde:
        #id numero de la pregunta
        #img es el recorte de la imagen original de una celda
        #coord son las coordenadas de la celda en cuestion. De la forma (x1,x2,y1,y2)
    """
    celdas = []
    celdas_izq = []
    celdas_der = []
    y1, y2, y3, y4 = i_cols[0], i_cols[1], i_cols[2], i_cols[3] 
    k = 4 #Para evitar lineas de las celdas
    n = 1
    for i in range(len(i_rows) - 1):
        x1 = i_rows[i]
        x2 = i_rows[i+1]

        celda1 = (n, img[x1+k:x2-k,y1+k:y2-k], (x1,x2,y1,y2))
        celda2 = (5 + n, img[x1+k:x2-k,y3+k:y4-k], (x1,x2,y1,y2))

        celdas_izq.append(celda1)
        celdas_der.append(celda2)
        n += 1

    celdas = celdas_izq + celdas_der

    #Visualizamos los avances:
    if ver_celdas:
        indices = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
        plt.figure()
        for i in range(len(celdas)):
            img = celdas[i][1]
            id = celdas[i][0]
            plt.subplot(5, 2, indices[i])
            plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Pregunta ' + str(id))
            plt.xticks([]), plt.yticks([])

        plt.suptitle('Preguntas del examen')
        plt.savefig('celdas.png')
        plt.show()

    return celdas


def obtener_respuestas(celdas, ver_bins = False, ver_rtas = False):
    """A partir de las celdas de un examen, almacena el numero de pregunta, crop
    y las coordenadas de cada respuesta en una lista y la devuelve como resultado.
    Como agregado, se puede especificar si se desea ver las imagen binarias
    y las imagenes de cada respuesta
    """

    """Lista en donde almaceno una estructura de la siguiente forma: (id,img,coord)
    Donde:
        #id numero de la pregunta
        #img es el recorte de la imagen original de una celda
        #coord son las coordenadas de la celda en cuestion. De la forma (x1,x2,y1,y2)
    """
    rtas = []
    for i, celda in enumerate(celdas):
        img = celdas[2][1]
        img = celda[1]
        img_bin = img < 40

        if ver_bins:
            plt.figure()
            ax = plt.subplot(121)
            plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Imagen original')
            plt.subplot(122, sharex=ax, sharey=ax)
            plt.imshow(img_bin, cmap='gray'), plt.title('Imagen + umbralado < 40')
            plt.savefig('Imagen+umbralado40.png')
            plt.show()

        
        #Sumamos la cantida de pixeles de cada fila
        sum_rows = np.sum(img_bin,1)

        #Obtenemos el indice de las linea horizontal con mas pixeles
        x2 = np.argmax(sum_rows)
        x1 = x2 - 14

        y1 = np.where(img_bin[x2,:])[0][0]
        y2 = np.where(img_bin[x2,:])[0][-1]

        id = i + 1
        img_rta = img[x1:x2,y1:y2]
        coord = (x1,x2,y1,y2)
        rta = (id, img_rta, coord)

        rtas.append(rta)
        
    if ver_rtas:
        plt.figure(figsize=(10, 10))
        for i, (id, img, coord) in enumerate(rtas):
            plt.subplot(5, 2, i + 1)
            plt.imshow(celdas[i][1], cmap='gray', vmin=0, vmax=255)  # Mostrar la celda original
            plt.imshow(img, cmap='gray', alpha=0.5)  # Superponer la imagen de respuesta
            plt.title(f'Respuesta a pregunta {id}')
            plt.xticks([]), plt.yticks([])

        plt.suptitle('Respuestas del examen')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('respuestas.png')
        plt.show()

    return rtas

def obtener_encabezado(img, ver_bin = False, ver_campos = False):
    """A partir de la imagen de un examen, almacena el nombre, crop y las
    coordenadas de cada campo del encabezado en una lista y la devuelve como resultado.
    Como agregado, se puede especificar si se desea ver la imagen binaria
    y las imagenes de cada celda
    """
    
    img_bin = img < 128 

    #Mostramos los cambios
    if ver_bin:
        plt.figure()
        ax = plt.subplot(121)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Imagen original')
        plt.subplot(122, sharex=ax, sharey=ax)
        plt.imshow(img_bin, cmap='gray'), plt.title('Imagen + umbralado < 128')
        plt.savefig('Imagen+umbralado128.png')
        plt.show()

    #Sumamos la cantida de pixeles de cada fila
    sum_rows = np.sum(img_bin,1)

    #Umbralamos el resultado para obtener los indices adecuados
    sum_rows_bin = sum_rows > 300 #Lineas de mas de 230 pixeles

    #Obtenemos los indices de las lineas horizontales
    
    i_rows = np.where(sum_rows_bin)[0]

    #El primer indice corresponde a la linea del encabezado
    x2 = i_rows[0]
    x1 = x2 - 22

    #Aplico diff a la imagen binaria en ese indice
    dif = np.diff(img_bin[x2])

    #Obtengo las coordenadas de las columnas (ys)
    i_cols = np.where(dif)

    i_cols = np.reshape(i_cols,(-1,2))

    """Lista en donde almaceno una estructura de la siguiente forma: (id,img,coord)
    Donde:
        #id nombre del campo
        #img es el recorte de la imagen original de un campo del encabezado
        #coord son las coordenadas del campo en cuestion. De la forma (x1,x2,y1,y2)
    """
    campos = []
    ids = ['Nombre', 'Fecha', 'Clase']
    for i, y in enumerate(i_cols):
        y1, y2 = y[0], y[1]
        campo = (ids[i], img[x1:x2,y1:y2], (x1,x2,y1,y2))
        campos.append(campo)

    #Visualizamos los avances:

    if ver_campos:
        plt.figure()
        for i in range(len(campos)):
            img = campos[i][1]
            id = campos[i][0]
            plt.subplot(1,3,i + 1)
            plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Campo ' + str(id))
            plt.xticks([]),plt.yticks([])
        plt.savefig('encabezado.png')
        plt.show()
        
    return campos

def obtiene_letra(rta, ver_bin = False, ver_letra = False):
    #Letra de tamaños fijos
    #Devuelve letra o '' si es vacia o contiene 2 letras

    #Convierto a imagen binaria
    rta_bin = np.uint8((rta < 129) * 255)
    if ver_bin:
        ax = plt.subplot(121)
        plt.imshow(rta, cmap='gray', vmin=0, vmax=255), plt.title('Imagen original')
        plt.subplot(122, sharex=ax, sharey=ax)
        plt.imshow(rta_bin, cmap='gray'), plt.title('Imagen + umbralado < 129')
        plt.savefig('Imagen+umbralado129.png')
        plt.show()
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(rta_bin, connectivity, cv2.CV_32S)
    
    cant = num_labels - 1
    if cant == 0 or cant > 1: #Si no tengo letras o si tengo mas de una letra
        return ''
    
    st = stats[1]
    x1, x2 = st[1], st[1] + st[3] 
    y1, y2 = st[0], st[0] + st[2]

    letra = rta_bin[x1:x2, y1:y2]
    if ver_letra:
        imshow(letra)
        plt.savefig('letra.png')
    contours, hierarchy = cv2.findContours(letra, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    cant_contor = hierarchy.shape[1]
    if cant_contor == 1:
        return 'C'
    elif cant_contor == 3:
        return 'B'
    elif cant_contor == 2 and contours[0].shape[0] < 15: #La 'A' tiene menos puntos que la 'D'
        return 'A'
    else:
        return 'D'
    

#PUNTO a)
def imprimir_correccion(respuestas, correctas):
    #Imprime el resultado de cada pregunta
    rtas = []
    for rta in respuestas:
        img = rta[1]
        letra = obtiene_letra(img)
        rtas.append(letra)
    
    correcion = []
    for i in range(len(rtas)):
        correcion.append(rtas[i] == correctas[i])

    for i,c in enumerate(correcion):
        if c:
            print("Pregunta " + str(i+1), 'OK')
        else:
            print("Pregunta " + str(i+1), 'MAL')

    aprobacion = sum(correcion) > 5
    return aprobacion


#PUNTO b)
def imprime_encabezado(encabezado, ver_bin = False):
    #Imprime la correcion del encabezado

    n = 'OK'
    f = 'OK'
    c = 'OK'

    nombre = encabezado[0][1]
    fecha = encabezado[1][1]
    clase = encabezado[2][1]

    #Convierto a imagen binaria
    nombre_bin = np.uint8((nombre < 129) * 255)
    fecha_bin = np.uint8((fecha < 140) * 255)
    clase_bin = np.uint8((clase < 129) * 255)

    if ver_bin:
        plt.subplot(321)
        plt.imshow(nombre, cmap='gray', vmin=0, vmax=255), plt.title('Imagen original')
        plt.subplot(322)
        plt.imshow(nombre_bin, cmap='gray'), plt.title('Imagen + umbralado < 129')
        plt.subplot(323)
        plt.imshow(fecha, cmap='gray', vmin=0, vmax=255), plt.title('Imagen original')
        plt.subplot(324)
        plt.imshow(fecha_bin, cmap='gray'), plt.title('Imagen + umbralado < 140')
        plt.subplot(325)
        plt.imshow(clase, cmap='gray', vmin=0, vmax=255), plt.title('Imagen original')
        plt.subplot(326)
        plt.imshow(clase_bin, cmap='gray'), plt.title('Imagen + umbralado < 129')
        plt.show()
    
    #Analizamos el campo nombre
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(nombre_bin, connectivity, cv2.CV_32S)
    cant_letras = num_labels - 1
    y1_letras = stats[1:,0]
    hay_espacio = (np.ediff1d(y1_letras) > 14).any()
    if cant_letras > 25 or not hay_espacio: #Mas de 25 letras o es una sola palabra
        n = 'MAL'
    
    #Analizamos el campo fecha
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fecha_bin, connectivity, cv2.CV_32S)
    cant_letras = num_labels - 1
    y1_letras = stats[1:,0]
    hay_espacio = (np.ediff1d(y1_letras) > 14).any()
    if cant_letras != 8 or hay_espacio: #No tiene 8 letras o no es una sola palabra
        f = 'MAL'

    #Analizamos el campo clase
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clase_bin, connectivity, cv2.CV_32S)
    cant_letras = num_labels - 1

    if cant_letras != 1:
        c = 'MAL'

    print('Nombre: ', n)
    print('Fecha: ', f)
    print('Clase: ', c)



"""a)Correcion de examen"""
img = cv2.imread(r'Imagenes de Entrada\examen_2.png', cv2.IMREAD_GRAYSCALE)
celdas = obtener_celdas(img)
respuestas = obtener_respuestas(celdas)

correctas = ['C','B','A','D','B','B','A','B','D','D']
print("Correcion de examen: \n")
imprimir_correccion(respuestas, correctas)


"""b)Correcion de encabezado"""
encabezado = obtener_encabezado(img)
print("Correcion de encabezado: \n")
imprime_encabezado(encabezado)


"""c)Aplico las funciones definidas anteriormente para cada examen"""
#Almaceno la aprobacion de cada examen para el item d)
#Guardo crop del campo nombre y True o False segun apruebe o no
resultados = [] 
for n in range(1,6):
    img = cv2.imread(r'Imagenes de Entrada\examen_' + str(n) +'.png', cv2.IMREAD_GRAYSCALE)

    celdas = obtener_celdas(img)
    respuestas = obtener_respuestas(celdas)
    encabezado = obtener_encabezado(img)

    
    #Correcion de encabezado
    print('\n-----Examen ' + str(n) + '-----\n')
    print("Encabezado:")
    print("----------------")
    imprime_encabezado(encabezado)
    print("----------------\n")
    #Correcion de examen
    #             1   2   3   4   5   6   7   8   9  10
    correctas = ['C','B','A','D','B','B','A','B','D','D']
    print("Correccion:")
    print("----------------")
    aprobacion = imprimir_correccion(respuestas, correctas)
    img_nombre = encabezado[0][1]

    resultados.append((img_nombre,aprobacion))
    print("----------------\n")


"""d)"""
#Creo una imagen artificial
img_nombre = resultados[0][0] #Imagen del nombre del primer examen
ancho_nombre = img_nombre.shape[1]
alto_nombre = img_nombre.shape[0]
ancho = ancho_nombre
alto = alto_nombre * len(resultados)

#Creo imagen blanco con ancho y alto calculados anteriormente
img_blanca = np.uint8(np.ones((alto,ancho)) * 255)
img_blanca.shape

#Recorto una letra A de la imagen del examen para especificar si aprobo o no
A = img[100:115,33:48]
ancho_A = A.shape[1]
alto_A = A.shape[0]

#Calculo ancho y alto de los crop de nombre
x1 = 0

#Coordenadas donde pegar la A en caso de aprobacion
x1A = 3
x2A = x1A + alto_A
y1A = - ancho_A - 5
y2A = y1A + ancho_A

img_salida = img_blanca.copy()
#Para cada uno de los resultados
for nombre in resultados:
    img_nombre = nombre[0].copy() #Imagen del nombre, copia para no modificar original
    aprobo = nombre[1]
    if aprobo: #Si aprobo le pego una A a la derecha del nombre 
        img_nombre[x1A:x2A,-20:-5] = A
    
    #Calculo ancho y alto del nombre
    ancho_nombre = img_nombre.shape[1]
    alto_nombre = img_nombre.shape[0]

    #Lo pego en la imagen de salida
    img_salida[x1:x1+alto_nombre,0:ancho] = img_nombre
    
    #Incremento la primer coordenada con el alto del nombre
    x1 += alto_nombre

#Muestro la imagen de salida resultante
imshow(img_salida, title='Resultados de los examenes', colorbar=False)
plt.savefig('Resultados de los examenes.png')