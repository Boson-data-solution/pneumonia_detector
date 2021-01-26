# Data processing config
LABELS = ['opacity', 'normal']
IMG_SIZE = 128

# Model training hyperparematers
HIDDEN_LAYER = 5
ACTIVATION = 'relu'
INPUT_LAYER_SIZE = 32
HIDDEN_LAYER_SIZE = 128
KERNEL_SIZE = (2, 2)
CONV_STRIDES = (1, 1)
POOL_STRIDES = (2, 2)
POOL_SIZE = (2, 2)
DROPOUT = 0.2
LOSS = 'categorical_crossentropy'
OPTIMIZER = 'adam'
METRICS = ['acc']

BATCH_SIZE = 10
EPOCHS = 10
