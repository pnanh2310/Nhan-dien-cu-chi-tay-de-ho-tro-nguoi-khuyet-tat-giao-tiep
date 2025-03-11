import tensorflow as tf
from generators import test_generator

def evaluate_model(model_path):
    model = tf.keras.models.load_model(model_path)
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy: {test_accuracy:.2f}')
    return test_loss, test_accuracy

if __name__ == '__main__':
    evaluate_model('E:\\BtlAI\\superAI.h5')
