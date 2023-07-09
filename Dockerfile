# Use uma imagem base apropriada do Python
FROM python:3.9

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o código Python para o diretório de trabalho do contêiner
COPY . /app

# Instale as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Especifique o comando a ser executado quando o contêiner for iniciado
CMD [ "python", consumer.py" ]
