from kafka import KafkaConsumer
from multiprocessing import Process, freeze_support
from ePred import fullAnalysis, iaTraining
from models import TaskOperationEnum
import json


# Configurações do consumidor Kafka
bootstrap_servers = 'localhost:9092'
topics = ['full_analysis', 'ia_train', 'topic3']

def consume_topic(topic):
    consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers)
    print("Iniciando consumidor para o tópico: " + topic)
    try:
        for message in consumer:
            message_value = message.value.decode('utf-8')
            message_object = json.loads(message_value)
            operation = message_object['operation']

            if(message.topic == "full_analysis" and operation == TaskOperationEnum.FULL_ANALYSIS):
                print("Eu consumi uma operação full_analysis")
                fullAnalysis(message_object['uuid'])
            elif(message.topic == "customized_analysis" and operation == TaskOperationEnum.CUSTOMIZED_ANALYSIS):
                print("Eu consumi uma operação customized_analysis")
            elif(message.topic == "ia_train" and operation == TaskOperationEnum.IA_TRAIN):
                print("Eu consumi uma operação ia_train")
                iaTraining(message_object['uuid'])
            else:
                print(f"Operação não reconhecida {message.topic}: {message_object}")
            
    except KeyboardInterrupt:
        pass

    finally:
        consumer.close()

if __name__ == '__main__':
    freeze_support()

    processes = []
    for topic in topics:
        print("Iniciando processo para o tópico: " + topic)
        p = Process(target=consume_topic, args=(topic,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
