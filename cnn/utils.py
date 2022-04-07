import os
import xml.etree.ElementTree as ET
from glob import glob

import cv2
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches

import base64
import numpy as np

mean1 = 0.485
mean2 = 0.456
mean3 = 0.406
std1  = 0.229
std2  = 0.224
std3  = 0.225

def default_transforms():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[mean1, mean2, mean3], std=[std1, std2, std3])])

def xml_to_csv(xml_folder, output_file=None):
    xml_list = []
    image_id = 0
    # Loop through every XML file
    for xml_file in glob(xml_folder + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Each object represents each actual image label
        for member in root.findall('object'):
            box = member.find('bndbox')
            label = member.find('name').text

            # Add image file name, image size, label, and box coordinates to CSV file
            row = (filename, width, height, label, int(float(box.find('xmin').text)),
                   int(float(box.find('ymin').text)), int(float(box.find('xmax').text)), int(float(box.find('ymax').text)), image_id)
            xml_list.append(row)
        
        image_id += 1

    # Save as a CSV file
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'image_id']
    xml_df = pd.DataFrame(xml_list, columns=column_names)

    if output_file is not None:
        xml_df.to_csv(output_file, index=None)

    return xml_df

def is_iterable(variable):
    return isinstance(variable, list) or isinstance(variable, tuple)

def read_image(path):
    if not os.path.isfile(path):
        raise ValueError(f'Could not read image {path}')

    with open(path, "rb") as img_file:
        '''
        add encode to base64 then decode in order to mimic the environment of bccaas
        '''
        base64_img = base64.b64encode(img_file.read()).decode('utf-8')
        buff = base64.b64decode(base64_img)
        im_np = np.frombuffer(buff, dtype=np.uint8)
        image =  cv2.imdecode(im_np, flags=1)
        return image

    # image = cv2.imread(path)

    # try:
    #     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # except cv2.error as e:
    #     raise ValueError(f'Could not convert image color: {str(e)}')

    # return rgb_image

def reverse_normalize(image):
    """Reverses the normalization applied on an image by the
    :func:`detecto.utils.reverse_normalize` transformation. The image
    must be a `torch.Tensor <https://pytorch.org/docs/stable/tensors.html>`_
    object.
    :param image: A normalized image.
    :type image: torch.Tensor
    :return: The image with the normalization undone.
    :rtype: torch.Tensor
    **Example**::
        >>> import matplotlib.pyplot as plt
        >>> from torchvision import transforms
        >>> from detecto.utils import read_image, \\
        >>>     default_transforms, reverse_normalize
        >>> image = read_image('image.jpg')
        >>> defaults = default_transforms()
        >>> image = defaults(image)
        >>> image = reverse_normalize(image)
        >>> image = transforms.ToPILImage()(image)
        >>> plt.imshow(image)
        >>> plt.show()
    """

    reverse = transforms.Normalize(mean=[-mean1 / std1, -mean2 / std2, -mean3 / 0.255],
                                   std=[1 / std1, 1 / std2, 1 / 0.255])
    return reverse(image)

def show_labeled_image(image, boxes, labels=None):
    """Show the image along with the specified boxes around detected objects.
    Also displays each box's label if a list of labels is provided.
    :param image: The image to plot. If the image is a normalized
        torch.Tensor object, it will automatically be reverse-normalized
        and converted to a PIL image for plotting.
    :type image: numpy.ndarray or torch.Tensor
    :param boxes: A torch tensor of size (N, 4) where N is the number
        of boxes to plot, or simply size 4 if N is 1.
    :type boxes: torch.Tensor
    :param labels: (Optional) A list of size N giving the labels of
            each box (labels[i] corresponds to boxes[i]). Defaults to None.
    :type labels: torch.Tensor or None
    **Example**::
        >>> from detecto.core import Model
        >>> from detecto.utils import read_image
        >>> from detecto.visualize import show_labeled_image
        >>> model = Model.load('model_weights.pth', ['tick', 'gate'])
        >>> image = read_image('image.jpg')
        >>> labels, boxes, scores = model.predict(image)
        >>> show_labeled_image(image, boxes, labels)
    """

    fig, ax = plt.subplots(1)
    # If the image is already a tensor, convert it back to a PILImage
    # and reverse normalize it
    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = transforms.ToPILImage()(image)
    ax.imshow(image)

    # Show a single box or multiple if provided
    if boxes.ndim == 1:
        boxes = boxes.view(1, 4)

    if labels is not None and not is_iterable(labels):
        labels = [labels]

    # Plot each box
    for i in range(boxes.shape[0]):
        box = boxes[i]
        width, height = (box[2] - box[0]).item(), (box[3] - box[1]).item()
        initial_pos = (box[0].item(), box[1].item())
        rect = patches.Rectangle(initial_pos,  width, height, linewidth=1,
                                 edgecolor='r', facecolor='none')
        if labels:
            ax.text(box[0] + 5, box[1] - 5, '{}'.format(labels[i]), color='red')

        ax.add_patch(rect)

    plt.show()

