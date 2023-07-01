from kafka import KafkaConsumer

print("abacate")
# Configurações do consumidor Kafka
bootstrap_servers = 'localhost:9092'
topic_name = 'full_analysis'

def consume_queue():
    consumer = KafkaConsumer(topic_name, bootstrap_servers=bootstrap_servers)

    try:
        i = 0
        while i < 10:
            i += 1
            print("abacate2")
            for message in consumer:
                message_value = message.value.decode('utf-8')
                print(message_value)
                print("Eu consumi a fila")
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

# Executa o consumidor
consume_queue()