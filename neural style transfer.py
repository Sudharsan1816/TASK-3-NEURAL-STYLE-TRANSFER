import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess image
def load_and_process_image(image_path, target_size=None):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = vgg19.preprocess_input(img_array)
    return img_array

# Deprocess the image to display
def deprocess_image(processed_img):
    img = processed_img.copy()
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# Compute content loss
def content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# Compute Gram matrix for style loss
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

# Compute style loss
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    return tf.reduce_mean(tf.square(S - C))

# Total variation loss for smoother output
def total_variation_loss(x):
    a = tf.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = tf.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

# Load VGG19 model
def get_model():
    vgg = vgg19.VGG19(weights='imagenet', include_top=False)
    vgg.trainable = False
    style_layer_names = [
        'block1_conv1', 'block2_conv1',
        'block3_conv1', 'block4_conv1',
        'block5_conv1'
    ]
    content_layer_name = 'block5_conv2'
    outputs = [vgg.get_layer(name).output for name in style_layer_names + [content_layer_name]]
    model = tf.keras.Model([vgg.input], outputs)
    return model, style_layer_names, content_layer_name

# Neural Style Transfer
def neural_style_transfer(content_path, style_path, output_path, iterations=1000, content_weight=1e3, style_weight=1e-2):
    model, style_layer_names, content_layer_name = get_model()
    
    content_image = load_and_process_image(content_path, target_size=(400, 400))
    style_image = load_and_process_image(style_path, target_size=(400, 400))
    generated_image = tf.Variable(content_image, dtype=tf.float32)

    optimizer = tf.optimizers.Adam(learning_rate=5.0)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            outputs = model(generated_image)
            style_outputs = outputs[:len(style_layer_names)]
            content_outputs = outputs[len(style_layer_names):]

            style_loss_value = 0
            for target_style, comb_style in zip(style_image_outputs, style_outputs):
                style_loss_value += style_loss(target_style, comb_style)
            style_loss_value *= style_weight / len(style_layer_names)

            content_loss_value = content_loss(content_outputs[0], target_content)
            content_loss_value *= content_weight

            loss = sStyle_loss_value + content_loss_value + total_variation_loss(generated_image)
        gradients = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(gradients, generated_image)])
        return loss

    target_content = model(content_image)[len(style_layer_names):]
    style_image_outputs = model(style_image)[:len(style_layer_names)]

    for i in range(iterations):
        loss = train_step()
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")
            img = deprocess_image(generated_image.numpy()[0])
            plt.imshow(img)
            plt.show()

    # Save final image
    final_img = deprocess_image(generated_image.numpy()[0])
    tf.keras.utils.save_img(output_path, final_img)
    print(f"Final image saved to {output_path}")
