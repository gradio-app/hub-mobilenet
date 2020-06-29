from imagenetlabels import idx_to_labels
import tensorflow as tf
from gradio import inputs, outputs

mobile_net = tf.keras.applications.MobileNetV2()

graph = tf.get_default_graph()
sess = tf.keras.backend.get_session()

def classify_image(inp):
    with graph.as_default():
        with sess.as_default():
            inp = inp.reshape((1, 224, 224, 3))
            prediction = mobile_net.predict(inp).flatten()
            return {idx_to_labels[i].split(',')[0]: float(prediction[i]) for i in range(1000)}


imagein = inputs.Image(shape=(224, 224, 3))
label = outputs.Label(num_top_classes=3)

gr.Interface(
    classify_image, 
    imagein, 
    label).launch();
