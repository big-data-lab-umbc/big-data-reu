tepochs=32000
def step(epoch, lr):
   lr = 1e-3
   if epoch >= 64:
      lr = 1e-4
   elif epoch >= 128:
      lr = 1e-5
   elif epoch >= 256:
      lr = 1e-6
   return lr
