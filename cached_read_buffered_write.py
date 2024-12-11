import numpy as np
from collections import deque

class CachedReaderBufferedWriter:
    def __init__(self, cache_size=100, buffer_size=10, cache_expiry=5):
        """
        Parent class with cached read and buffered write protocol.
        
        :param cache_size: Maximum number of cache entries.
        :param buffer_size: Maximum size of the write buffer.
        :param cache_expiry: Number of reads after which a cache entry expires.
        """
        self.cache = {}
        self.cache_size = cache_size
        self.write_buffer = deque(maxlen=buffer_size)
        self.cache_expiry = cache_expiry  # Cache entry expiry after k reads
        self.child_class = None  # This will hold the child neural network instance
        

    def set_child(self, child_instance):
        """Attach a child class instance."""
        self.child_class = child_instance
        #self.table = self.child_class.table

    def query(self, input_data):
        """Query the cache or the child class for a prediction."""
        input_data = np.array(input_data)
        if(input_data.shape[0] > 1):
            # print('*******************************************************')
            # print('querying multiple samples not supported', input_data.shape[0])
            # print(input_data)
            # exit()
            return self.child_class.predict(input_data)

        if(input_data.shape[0] == 1):
            input_data_ = input_data[0]

        input_tuple = tuple(input_data_)
        if input_tuple in self.cache:
            # Increment the read counter
            self.cache[input_tuple]['reads'] += 1
            
            # Check if the cache entry has expired
            if self.cache[input_tuple]['reads'] > self.cache_expiry:
                # Refresh the cache entry
                result = self.child_class.predict(input_data)
                self.cache[input_tuple] = {'value': result, 'reads': 1}
            else:
                # Return the cached value
                result = self.cache[input_tuple]['value']
        else:
            # Compute the result using the child class
            result = self.child_class.predict(input_data)
            # Add the result to the cache
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))  # Remove the oldest cached item
            self.cache[input_tuple] = {'value': result, 'reads': 1}
        return result

    def update(self, input_data, target):
        """Buffer updates and perform batched updates when buffer is full."""
        self.write_buffer.append((input_data, target))
        if len(self.write_buffer) == self.write_buffer.maxlen:
            # Perform a batched update in the child class
            inputs, targets = zip(*self.write_buffer)
            self.child_class.train(np.array(inputs), np.array(targets))
            self.write_buffer.clear()  # Clear the buffer after the update
    
    def predict(self, input_data):
        return self.query(input_data)

    def train(self, input_data, target, num_epochs = 1, bookkeep_train = False):
        self.update(input_data, target)
