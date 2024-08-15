import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import backend as K
from . import utils

class getMIOutput(tf.keras.callbacks.Callback):
    def __init__(self, trn, tst, Z_layer_idx, num_selection, do_save_func=None, *kargs, **kwargs):
        super(getMIOutput, self).__init__(*kargs, **kwargs)
        self.trn = trn
        self.tst = tst
        self.Z_layer_idx = Z_layer_idx
        self.num_selection = num_selection
        self.do_save_func = do_save_func # control the saved epoch
        self.layer_values = []
        self.layerixs = []
        self.layerfuncs = []

    def on_train_begin(self, logs=None):
        for lndx, l in enumerate(self.model.layers):
            self.layerixs.append(lndx)
            self.layer_values.append(lndx)
            self.layerfuncs.append(K.function(self.model.inputs, [l.output,]))

    def on_epoch_end(self, epoch, logs=None):
        if self.do_save_func is not None and not self.do_save_func(epoch):
            return

        data = {
            'activity': []  # Activity in each layer
        }

        for lndx, layerix in enumerate(self.layerixs):
            if lndx == self.Z_layer_idx:
                clayer = self.model.layers[layerix]
                activity_tst = self.layerfuncs[lndx]([self.trn[:self.num_selection],])[0]
                data['activity'].append(activity_tst)

        # Convert the list of numpy arrays to a single numpy array for npy compatibility
        activity_tst_array = np.array(data['activity']).reshape(self.num_selection, -1)
        
        # Save the numpy array to an npy file
        filename = f"IB_epoch_{epoch}_z_{self.Z_layer_idx}.npy"
        filepath = os.path.join('savedata', filename)
        np.save(filepath, activity_tst_array)

        print(f"Saved data for epoch {epoch} to {filename}")
        
        
def do_report_IB(epoch):
    # Only log activity for some epochs.  Mainly this is to make things run faster.
    if epoch < 20:       # Log for all first 20 epochs
        return True
    elif epoch < 100:    # Then for every 5th epoch
        return (epoch % 5 == 0)
    elif epoch < 2000:    # Then every 10th
        return (epoch % 20 == 0)
    else:                # Then every 100th
        return (epoch % 100 == 0)
    
    

def train_model(config):
    # Get data
    trn, tst = utils.get_IB_data('2017_12_21_16_51_3_275766')

    # Model training
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    input_layer = tf.keras.layers.Input((trn.X.shape[1],))
    x = tf.keras.layers.Dense(10, activation='tanh')(input_layer)
    x = tf.keras.layers.Dense(7, activation='tanh')(x)
    x = tf.keras.layers.Dense(5, activation='tanh')(x)
    x = tf.keras.layers.Dense(4, activation='tanh')(x)
    x = tf.keras.layers.Dense(3, activation='tanh')(x)
    CE_output = tf.keras.layers.Dense(2, activation='softmax', name='CE')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=[CE_output])

    # Use the optimizer and learning rate from the config
    if config["optimizer"] == "SGD":
        opt = tf.keras.optimizers.SGD(learning_rate=config["lr"])
    elif config["optimizer"] == "Adam":
        opt = tf.keras.optimizers.Adam(learning_rate=config["lr"])
    # Add other optimizers as needed

    model.compile(optimizer=opt,
                  loss={'CE': 'categorical_crossentropy'},
                  metrics={'CE': 'accuracy'})

    reporter = getMIOutput(trn=trn.X,
                           tst=tst.X,
                           Z_layer_idx=config["z_idx"],  # Use z_idx from config
                           num_selection=trn.X.shape[0],
                           do_save_func=do_report_IB)

    history = model.fit(x=trn.X, y=trn.Y,
                        batch_size=config["batch_size"],  # Use batch size from config
                        epochs=config["epoch"],  # Use number of epochs from config
                        verbose=0,
                        validation_data=(tst.X, tst.Y),
                        callbacks=[reporter,])

    # Print the final generalization gap (train accuracy - test accuracy / train loss - test loss)
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    generalization_gap_acc = final_train_acc - final_val_acc
    generalization_gap_loss = final_train_loss - final_val_loss
    print(f"Final train (Accuracy): {final_train_acc}")
    print(f"Final test (Accuracy): {final_val_acc}")
    print(f"Final Generalization Gap (Accuracy): {generalization_gap_acc}")
    print(f"Final Generalization Gap (Loss): {generalization_gap_loss}")
