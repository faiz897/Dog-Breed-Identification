# for getting our input ready
import tensorflow as tf

# for our model ready
import tensorflow_hub as Hub

# for getting our outputs ready
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    '''
    Takes the image_path as input and process on it and return the processed image
    :param image_path: image_path
    :return: processed_image
    '''
    # read the image
    image = tf.io.read_file(image_path)
    # decode the image and turn into tensors
    image = tf.io.decode_jpeg(image)
    # normalize our image color channels (red, blue, green) values from 0-255 to 0-1
    image = tf.image.convert_image_dtype(image, tf.float32)
    # reshape our image according to our trained model
    image = tf.image.resize(image, (224, 224))

    # return our processed image
    return image

def show_image(image_path, predicted_label):
    '''
    Takes the image file path as input, turn the image into processed image and show it
    :param image_path: image path
    :return: show the image
    '''
    image = process_image(image_path=image_path)
    # setup the image size
    plt.figure(figsize=(7, 5))
    # show the input image
    plt.imshow(image)
    # turn the grid lines off
    plt.xticks([])
    plt.yticks([])
    # add the title
    plt.title(f"Dog Breed Name: {predicted_label}")
    # show the image
    plt.show()

# turn our test image into batch
def create_data(image_path, batch_size=32):
    '''
    takes a processed image as input, create data batch and return the batched data set
    :param image: processed_image
    :param batch_size: 32 (default)
    :return: batched_dataset
    '''
    x = []
    # image = show_inp_image(image_path)
    x.append(image_path)
    print("create the input databatch.................")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    data_batch = data.map(process_image).batch(batch_size=batch_size)
    return data_batch

# load our model
def load_model(model_path):
    '''
    takes the trained model path and return the trained model
    :param model_path: trained_model_path
    :return: trained_model
    '''
    print(f"load model from : {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": Hub.KerasLayer})
    print("model loaded successfully proceed.....!")
    return model

# show the output
def output(prediction_probabilities, input_imagepath):
    '''
    takes model predictions and convert them to an understandable form
    :param model_predictions: model predictions on input image
    :param input_imagepath: input image path
    :return: a proper figure with image and their predicted label
    '''
    # import the breeds name which model predicts
    labels_csv = pd.read_csv("breeds.csv")
    labels = labels_csv["breeds"].to_numpy()

    # getting the predicted label
    predicted_label = labels[np.argmax(prediction_probabilities)]
    print(f"Breed of the input dog is: {predicted_label}")

    # visualize our input image and predicted label
    show_image(image_path=input_imagepath, predicted_label=predicted_label)


if __name__ == '__main__':
    image_path = input("Enter the image path :" )
    image = create_data(image_path=image_path)
    model = load_model(model_path="Model/20212605/01/21-152612-Full-model-mobilenetv2-Adam")
    model_predictions = model.predict(image)
    output(prediction_probabilities=model_predictions, input_imagepath=image_path)