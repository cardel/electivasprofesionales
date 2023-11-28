"""
Aplicación en Flask para modelo de Redes Neuronales
"""

from flask import Flask, request # Importar Flask y request
from flask_restful import Resource, Api # Importar Resource y Api
import keras
import pickle
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__) # Instancia de Flask
api = Api(app) # Instancia de Api()

#Cargar los modelos
modelo = keras.models.load_model('modeloprediccion')
pipedato = pickle.load(open("pipelinedatos.sav","rb"))

class intro(Resource):
    """
    Función para la ruta raíz
    """
    def get(self):
        """
        Método GET
        """
        return {'message': 'Bienvenido a la API de Redes Neuronales'}, 200

    def post(self):
        """
        Método POST
        """
        return {'message': 'Bienvenido a la API de Redes Neuronales'}, 200

class leer(Resource):
    """
    Función para la ruta /leer
    """
    def get(self):
        """
        Método GET
        """
        dato = request.args.get('dato') # Obtener el dato
        if dato is None:
            return {'message': 'No se ha enviado ningún dato'}, 400
        else:
            return {'message': 'Dato recibido: {}'.format(dato)}, 200

    def post(self):
        """
        Método POST
        """
        dato = request.form['dato']
        if dato is None:
            return {'message': 'No se ha enviado ningún dato'}, 400
        else:
            return {'message': 'Dato recibido: {}'.format(dato)}, 200

class predict(Resource):
    def post(selft):
        file = request.files['imagen']
        if file is None:
            return {'message': 'No se ha enviado ningún archivo'}, 400
        else:
            img = plt.imread(file) #Leer la imagen con matplotlib
            #img = pipedato.transform(img) en caso de tener un pipelinedatos
            img = rgb2gray(img)
            img = resize(img, (28,28))
            res = modelo.predict(img.reshape(1,28,28,1))
            return {'result': f'El número es: {np.argmax(res[0])}',
                    'total': f'El total de predicciones es {res} '}, 200

api.add_resource(intro, '/') # Ruta raíz
api.add_resource(leer, '/leer') # Ruta /leer
api.add_resource(predict, '/predict') # Ruta /predict

if __name__ == '__main__':
    app.run(debug=True) # Ejecutar en modo de desarrollo


