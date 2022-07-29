tepochs=32000
def step(epoch, lr):
   if epoch <= 64:
      return 1e-3
   elif epoch <= 128:
      return 1e-4
   elif epoch <= 256:
      return 1e-5
   else:
      return 1e-6
