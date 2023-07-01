from kafka import KafkaConsumer
from multiprocessing import Process, freeze_support
from FullAnalysis import analyse


# Configurações do consumidor Kafka
bootstrap_servers = 'localhost:9092'
topics = ['full_analysis', 'topic2', 'topic3']

def consume_topic(topic):
    consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers)

    try:
        for message in consumer:
            message_value = message.value.decode('utf-8')
            print(f"Eu consumi a fila {message.topic}: {message_value}")
            analyse()
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

if __name__ == '__main__':
    freeze_support()

    processes = []
    for topic in topics:
        p = Process(target=consume_topic, args=(topic,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
