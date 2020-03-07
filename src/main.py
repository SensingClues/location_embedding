# main.py
from data.dataset import *
from model.network import get_model
import tensorflow_addons as tfa



def main():
    df = get_files_from_path(pathstring="data/tiles/", zoom=14)

    train_dataset = get_dataset(df)
    # val_dataset = ...

    # Get model
    model = get_model()
    # # #
    callbacks = [
        # tf.keras.callbacks.ModelCheckpoint('models/checkpoints/model', monitor='val_loss', verbose=0, save_best_only=True,
        tf.keras.callbacks.ModelCheckpoint('models/checkpoints/model', monitor='loss', verbose=0, save_best_only=True,
                                                     save_weights_only=True, mode='auto', period=1),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tfa.losses.TripletSemiHardLoss())

    history = model.fit(
        train_dataset,
        callbacks = callbacks,
        epochs=50)

if __name__=='__main__':
    main()