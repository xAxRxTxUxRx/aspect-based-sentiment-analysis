import tensorflow.keras.optimizers as opt


def get_optimizer(algorithm):
    clipvalue = 0
    clipnorm = 10

    if algorithm == 'rmsprop':
        optimizer = opt.RMSprop(learning_rate=0.1, epsilon=1e-06, clipnorm=clipnorm)
    elif algorithm == 'sgd':
        optimizer = opt.SGD(learning_rate=0.1, momentum=0.0, decay=0.0, nesterov=False, clipnorm=clipnorm)
    elif algorithm == 'adagrad':
        optimizer = opt.Adagrad(learning_rate=0.01, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
    elif algorithm == 'adadelta':
        optimizer = opt.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
    elif algorithm == 'adam':
        optimizer = opt.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm)
    elif algorithm == 'adamax':
        optimizer = opt.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm,
                               clipvalue=clipvalue)

    return optimizer