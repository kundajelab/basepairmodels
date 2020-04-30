from keras.callbacks import Callback
import time
import datetime

class BatchController(Callback):
    """Callback to trigger batch generation for next epoch
       
    """
    
    def __init__(self, mtbatchgenerator):
        # call base class constructor
        super(BatchController, self).__init__()
        
        # pointer to the batch generator
        self.mtbatchgenerator = mtbatchgenerator
    
    
    def on_epoch_begin(self, epoch, logs={}):
        # set ready flag in the batch generator to True
        self.mtbatchgenerator.set_ready_for_next_epoch(True)

        
class TimeHistory(Callback):
    """Callback to record start, end & elapsed time for each epoch
    
    """
    
    def on_train_begin(self, logs={}):
        # initialize
        self.times = []

        
    def on_epoch_begin(self, epoch, logs={}):
        # new dictionary
        self.times.append({})
        
        # record start times
        self.epoch_time_start = time.time()
        self.times[epoch]['start'] = str(datetime.datetime.now()) 

        
    def on_epoch_end(self, epoch, logs={}):
        # record end time
        self.times[epoch]['end'] = str(datetime.datetime.now()) 
        
        # compute elapsed time
        self.times[epoch]['elapsed'] = (time.time() - self.epoch_time_start)