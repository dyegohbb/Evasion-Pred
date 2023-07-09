# Use uma imagem base apropriada do Python
FROM python:3.9

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o código Python para o diretório de trabalho do contêiner
COPY . /app

ENV KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
    DB_USERNAME=root \
    DB_PASSWORD=QweBHU* \
    DB_HOST=localhost \
    DB_PORT=3306 \
    DB_NAME=evasionwatch \

# Instale as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Especifique o comando a ser executado quando o contêiner for iniciado
CMD [ "python", consumer.py" ]
