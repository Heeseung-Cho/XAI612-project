import numpy as np
import os
import argparse
import tensorflow as tf
from utils import dataloader, CustomStopper, CustomCheckpoint
import pickle
from glob import glob
from model import ResNet50, ResNet101
import pandas as pd
import wandb

## Set GPU
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'True'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

## Argparse
parser = argparse.ArgumentParser(description='XAI612-Project')

parser.add_argument('--epochs', default=300, type=int,
                    help='Epochs')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch Size')
parser.add_argument('--image_size', default=224, type=int,
                    help='Image Size')
parser.add_argument('--train_path', default="retina1_trainvalid/trainvalid", type=str,
                    help='Set source(train) path')
parser.add_argument('--test_path', default="test_big", type=str,
                    help='Set test path')
parser.add_argument('--model', default="resnet50", type=str,
                    help='Set test path')

def main():
  global args
  args = parser.parse_args()

  #Initial params
  trainvalid=args.train_path
  test=args.test_path
  root_path = os.getcwd()
  batch_size = args.batch_size
  img_height = args.image_size
  img_width = args.image_size
  save_path = root_path + '/saved_model'

  train_ds, val_ds, test_ds = dataloader(trainvalid, test, batch_size = batch_size, img_height = img_height, img_width = img_width)

  normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
  augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal"), ## Horizontal Flip
    tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.15, 0.15)), ## Random Rotation
    tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(0.0, 0.3),width_factor=(0.0, 0.3)), ## Random Zoom
    normalization_layer,  
  ]
  )

  train_ds = train_ds.map(lambda x, y: (augmentation_layer(x), y))
  val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
  test_ds = test_ds.map(lambda x: normalization_layer(x))

  ## Model
  model = ResNet101(h = img_height, w = img_width, c = 3)
  initial_learning_rate = 0.001
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate,
      decay_steps=10000,
      decay_rate=0.96,
      staircase=True)  
  optimizer = tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum = 0.9, decay = 0.0005)
  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
  modelPath = os.path.join(save_path, 'bestModel.h5')

  checkpoint = CustomCheckpoint( # set model saving checkpoints
      modelPath, # set path to save model weights
      monitor='val_loss', # set monitor metrics
      verbose=1, # set training verbosity
      save_best_only=True, # set if want to save only best weights
      save_weights_only=True, # set if you want to save only model weights
      mode='auto', # set if save min or max in metrics
      period=1, # interval between checkpoints
      start_epoch=100 # Checkpoint after start_epochs
      )

  earlystopping = CustomStopper(
    monitor='val_loss', # set monitor metrics
    min_delta=0.001, # set minimum metrics delta
    patience=30, # number of epochs to stop training
    restore_best_weights=True, # set if use best weights or last weights
    start_epoch=100 # Check patience after start_epochs
    )  
        
  callbacksList = [checkpoint,earlystopping] # build callbacks list

  ## Train
  model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy'])

  hist = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=callbacksList
  )
  with open(os.path.join(save_path, "hist.pkl"), "wb") as file:
      pickle.dump(hist.history, file)

  model.save(save_path)

  ## Test
  del model # Delete the original model, just to be sure!
  model = tf.keras.models.load_model(save_path, custom_objects={"CustomModel": ResNet101})
  prob = model.predict(test_ds)
  prediction = np.argmax(prob, axis = 1)
  print(prediction)

  submission = pd.DataFrame({
    'image-name':sorted(glob(test+'/0/*.*')),
    '<predicted label>':prediction
  })
  submission.to_csv(root_path+'/submission.csv',index=False)


if __name__ == "__main__":
    main()