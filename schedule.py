def step(epoch, lr):
    if epoch <= 512:
        return 1e-4
    elif epoch <= 768:
        return 1e-5
    elif epoch <= 1024:
        return 1e-6
    else: # maybe?
        return 1e-6
