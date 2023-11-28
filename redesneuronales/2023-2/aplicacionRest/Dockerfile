FROM python:3.11.6

WORKDIR /app
COPY . /app
RUN pip install --upgrade pip #Actualiza el pip interno
RUN pip install --no-cache-dir -r requirements.txt #Instala los requerimientos del archivo requirements.txt
EXPOSE 5000 
#Expone el puerto 5000
RUN chmod +x entrypoint.sh #Permite que el archivo entrypoint.sh se ejecute
CMD ["bash", "entrypoint.sh"]
