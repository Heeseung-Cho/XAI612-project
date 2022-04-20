import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
def dataloader(trainvalidpath, testpath, batch_size = 16, img_height = 224, img_width = 224):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    trainvalidpath,
    validation_split=0.2,
    subset="training",
    seed=42,
    shuffle = True,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    trainvalidpath,
    validation_split=0.2,
    subset="validation",
    seed=42,
    shuffle = False,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    testpath,
    label_mode = None,
    image_size=(img_height, img_width),
    shuffle = False
    )
    return train_ds, val_ds, test_ds


## Custom stopper with start epoch
class CustomStopper(EarlyStopping):
    def __init__(self, start_epoch, **kwargs): # add argument for starting epoch
        super(CustomStopper, self).__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            super().on_epoch_end(epoch, logs)

## Custom stopper with start epoch
class CustomCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, start_epoch, **kwargs): # add argument for starting epoch
        super(CustomCheckpoint, self).__init__(filepath,**kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            super().on_epoch_end(epoch, logs)

if __name__ == "__main__":
    pass
