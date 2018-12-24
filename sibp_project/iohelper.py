import cv2
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def read_and_resize_image(file_path, ROWS, COLS):
  img = cv2.imread(file_path, cv2.IMREAD_COLOR)
  return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def show_loss_plot(loss, val_loss, nb_epoch):
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('VGG-16 Loss Trend')
  plt.plot(loss, 'blue', label='Training Loss')
  plt.plot(val_loss, 'green', label='Validation Loss')
  plt.xticks(range(0,nb_epoch)[0::2])
  plt.legend()
  plt.show()

def show_image_plot(image):
  plt.imshow(image)
  plt.show()
