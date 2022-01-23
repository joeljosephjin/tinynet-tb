from tensorflow.python.framework import ops

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from train_loop import augment

import numpy as np
import sys
import os
import cv2
import argparse

def register_gradient():
    """
    register gradients for ReLU
    """
    if "GuidedBackPropReLU" not in ops._gradient_registry._registry:
        @tf.RegisterGradient("GuidedBackPropReLU")
        def _GuidedBackPropReLU(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)

    if "DeconvReLU" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("DeconvReLU")
        def _DeconvReLU(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype)


def everride_relu(name, images, path_vgg):
    """
    override ReLU for guided backpropagation
    """
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):
        model = vgg16.Vgg16(vgg16_npy_path=path_vgg)
        with tf.name_scope("content_vgg_gbp"):
            model.build(images)
    return model


def saliency_by_class(input_model, images, category_index, nb_classes=1000):
    """
    calculate d prob_of_class / d image
    """
    loss = tf.multiply(input_model.prob, tf.one_hot([category_index], nb_classes))
    reduced_loss = tf.reduce_sum(loss, axis=1)
    grads = tf.gradients(reduced_loss, images)
    return grads


def grad_cam(prob, category_index, layer_name, sess, feed_dict, nb_classes = 2):
    """
    calculate Grad-CAM
    """
    loss = tf.multiply(prob, tf.one_hot([category_index], nb_classes))
    reduced_loss = tf.reduce_sum(loss[0])
    conv_layer = sess.graph.get_tensor_by_name(layer_name + ':0')
    conv_layer_out = sess.graph.get_tensor_by_name('last_layer' + ':0')
    conv_gap_layer_out = sess.graph.get_tensor_by_name('gap' + ':0')
    # logits_layer_out = sess.graph.get_tensor_by_name('logits_layer' + ':0')
    # print('conv_layer.shape:', conv_layer.shape)
    # print('conv_layer_out.shape:', conv_layer_out.shape)
    grads = tf.gradients(reduced_loss, conv_layer)[0] # d loss / d conv

    # conv_layer_val, grads_val, conv_layer_out_val, conv_gap_layer_out_val, logits_layer_out_val = sess.run([conv_layer, grads, conv_layer_out, conv_gap_layer_out, logits_layer_out], feed_dict=feed_dict)
    conv_layer_val, grads_val, conv_layer_out_val, conv_gap_layer_out_val = sess.run([conv_layer, grads, conv_layer_out, conv_gap_layer_out], feed_dict=feed_dict)
    print('conv_layer.shape:', conv_layer_val.shape)
    print('conv_layer_out_val.shape:', conv_layer_out_val.shape)
    print('conv_gap_layer_out_val.shape:', conv_gap_layer_out_val.shape)
    # print('logits_layer_out_val.shape:', logits_layer_out_val.shape)
    print('grads_val (grads of loss/conv_layer):', grads_val.shape)
    
    grads_mean = np.mean(grads_val, axis=(1, 2)) # average pooling

    # cams = np.sum(weights * output, axis=3)
    # cams = np.sum(grads_val * output, axis=3)
    # cams = grads_mean * output
    for i in range(80):
        conv_layer_out_val[0,:,:,0] += conv_gap_layer_out_val[0, i] * conv_layer_out_val[0,:,:,i]
    cams = conv_layer_out_val[0,:,:,0]
    return cams
    # return cams[0,0,:,:, np.newaxis]
    # return output[0,0,:,:, np.newaxis]


def save_cam(cams, rank, class_id, class_name, prob, image_batch, input_image_path):
    """
    save Grad-CAM images
    """
    # cam = cams[0] # the first GRAD-CAM for the first image in  batch
    cam = cams # the first GRAD-CAM for the first image in  batch
    image = np.uint8(image_batch[0][:, :, ::-1] * 255.0) # RGB -> BGR
    # cam = cv2.resize(cam, (224, 224)) # enlarge heatmap
    print('cam.size:', cam.shape)
    cam = cv2.resize(cam, (512, 512)) # enlarge heatmap
    cv2.imwrite('cam.jpg', cam)
    cv2.imwrite('image.jpg', image)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam) # normalize
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) # balck-and-white to color
    cam = np.float32(cam) + np.float32(image) # everlay heatmap onto the image
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)

    # create image file names
    base_path, ext = os.path.splitext(input_image_path)
    base_path_class = "{}_{}_{}_{}_{}".format(base_path, rank, class_id, class_name, 'prob')
    cam_path = "{}_{}{}".format(base_path_class, "gradcam", ext)
    heatmap_path = "{}_{}{}".format(base_path_class, "heatmap", ext)
    segmentation_path = "{}_{}{}".format(base_path_class, "segmented", ext)

    # write images
    cv2.imwrite(cam_path, cam)
    cv2.imwrite(heatmap_path, (heatmap * 255.0).astype(np.uint8))
    cv2.imwrite(segmentation_path, (heatmap[:, :, None].astype(float) * image).astype(np.uint8))


def get_info(prob, file_path, top_n=5):
    """
    returns top_n information(class id, class name, probability and synset data
    """
    synsets = [l.strip() for l in open(file_path).readlines()]
    preds = np.argsort(prob)[::-1]

    top_n_synset = []
    for i in range(top_n):
        pred = preds[i]
        synset = synsets[pred]
        class_name = "_".join(synset.split(",")[0].split(" ")[1:])
        top_n_synset.append( (pred, class_name, prob[pred], synset) )
    return top_n_synset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', type=str, default='/path/to/image', help='path to image.')
    parser.add_argument('vgg16_path', type=str, default='/path/to/vgg16.npy', help='path to vgg16.npy.')
    parser.add_argument('--top_n', type=int, default=3, help="Grad-CAM for top N predicted classes.")
    args = parser.parse_args()
    print(args)

    size=512

    # input_image = utils.load_image(args.input_image) # tf RGB
    # image_batch = input_image[None, :, :, :3]

    batch_imgs = cv2.imread(args.input_image)[:, :, [0]]
    batch_labs = [0]

    # print('batch_imgs.size (original)',batch_imgs.shape)

    # batch_imgs, batch_labs = augment(batch_imgs, batch_labs, size)
    batch_imgs = cv2.resize(batch_imgs, (size, size))[..., np.newaxis]

    cv2.imwrite('images0.jpg', batch_imgs)

    # print('batch_imgs.size', batch_imgs.shape)

    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    # with tf.device('/cpu:0'):
    #     images = tf.placeholder("float", [None, 224, 224, 3])
    #     model = vgg16.Vgg16(vgg16_npy_path=args.vgg16_path)
    #     with tf.name_scope("content_vgg"):
    #         model.build(images)

    # path_synset = os.path.join(os.path.dirname(vgg16.__file__), "synset.txt")

    # load the model
    new_saver = tf.train.import_meta_graph('models/modelname.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('models'))

    # load the probability tensor and feed it here
    prob = graph.get_tensor_by_name('probabilities:0')
    # prob_val = sess.run(prob, {'input:0': [batch_imgs], 'labels:0': batch_labs})

    print('prob:', prob)

    # infos = get_info(prob[0], path_synset, top_n=args.top_n)
    # for rank, info in enumerate(infos):
    #     print("{}: class id: {}, class name: {}, probability: {:.3f}, synset: {}".format(rank, *info))

    # GRAD-CAM
    # for i in range(args.top_n):
    for i in range(1):
        class_id = 0
        class_name = "tb"
        model=None
        # cams = grad_cam(prob_val, class_id, "content_vgg/conv5_3/Relu", sess, feed_dict={{'input:0': [batch_imgs], 'labels:0': batch_labs}})
        cams = grad_cam(prob, class_id, 'conv2d_11/kernel', sess, feed_dict={'input:0': [batch_imgs], 'labels:0': batch_labs})

        print('cams.shape:', cams.shape)
        print('cams:', cams)

        save_cam(cams, i, class_id, class_name, prob, [batch_imgs], args.input_image)

    # Guided Backpropagation
    # register_gradient()

    # del model
    # images = tf.placeholder("float", [None, 224, 224, 3])

    # guided_model = everride_relu('GuidedBackPropReLU', images, args.vgg16_path)
    # class_id = infos[0][0]
    # class_saliencies = saliency_by_class(guided_model, images, class_id, nb_classes=1000)
    # class_saliency = sess.run(class_saliencies, feed_dict={images: image_batch})[0][0]

    # class_saliency = class_saliency - class_saliency.min()
    # class_saliency = class_saliency / class_saliency.max() * 255.0
    # base_path, ext = os.path.splitext(args.input_image)
    # gbprop_path = "{}_{}{}".format(base_path, "guided_bprop", ext)
    # cv2.imwrite(gbprop_path, class_saliency.astype(np.uint8))

if __name__ == '__main__':
    main()