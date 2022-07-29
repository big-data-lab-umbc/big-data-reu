tepochs = 4096
def step(epoch, lr):
    if epoch <= tepochs*(1./3):
    # if epoch <= 32:
        return 1e-3
    # elif epoch <= 768:
    elif epoch <= tepochs*(2./3):
    # if epoch <= 768:
        return 1e-4
    elif epoch <= tepochs*(5./6):
        return 1e-5
    else:
        return 1e-6
