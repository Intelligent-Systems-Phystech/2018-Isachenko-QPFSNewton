import numpy as np

def sgd_momentum(x, dx, config, state):
    
    # x and dx have complex structure, old dx will be stored in a simpler one
    state.setdefault('old_grad', {})
    
    i = 0 
    for cur_layer_x, cur_layer_dx in zip(x, dx): 
        for cur_x, cur_dx in zip(cur_layer_x, cur_layer_dx):
            cur_old_grad = state['old_grad'].setdefault(i, np.zeros_like(cur_dx))
            np.add(config['momentum'] * cur_old_grad, config['learning_rate'] * cur_dx, 
                   out = cur_old_grad)
            cur_x -= cur_old_grad
            i += 1
            
def nesterov(x, dx, config, state):
    state.setdefault('old_grad', {})

    i = 0 
    for cur_layer_x, cur_layer_dx in zip(x, dx): 
        for cur_x, cur_dx in zip(cur_layer_x, cur_layer_dx):
            
            cur_old_grad = state['old_grad'].setdefault(i, np.zeros_like(cur_dx))
            
            np.add(config['momentum'] * cur_old_grad, config['learning_rate'] * cur_dx, 
                   out = cur_old_grad)
            
            cur_x -= cur_old_grad * config['momentum'] + config['learning_rate'] * cur_dx
            i += 1
            
def adam(x, dx, config, state):
    state.setdefault('old_grad', {})
    state.setdefault('old_squared_grad', {})
    
    i = 0 
    for cur_layer_x, cur_layer_dx in zip(x, dx): 
        for cur_x, cur_dx in zip(cur_layer_x, cur_layer_dx):
            cur_old_grad = state['old_grad'].setdefault(i, np.zeros_like(cur_dx))
            cur_old_squared_grad = state['old_squared_grad'].setdefault(i, 
                                                                        np.zeros_like(cur_dx))
            
            np.add(config['momentum'] * cur_old_grad, (1 - config['momentum']) * cur_dx, 
                   out = cur_old_grad)
            
            np.add(config['squared_momentum'] * cur_old_squared_grad, 
                   (1 - config['squared_momentum']) * cur_dx ** 2,
                   out = cur_old_squared_grad)
            
            cog = cur_old_grad / (1 - config['momentum'])
            cosg = cur_old_squared_grad / (1 - config['squared_momentum'])
            
            cur_x -= config['learning_rate'] * np.divide(cog, np.sqrt(cosg) + config['eps'])
            i += 1