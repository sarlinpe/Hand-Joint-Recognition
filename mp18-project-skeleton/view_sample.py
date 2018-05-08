import h5py
import numpy
import matplotlib.pyplot as plt

f = h5py.File('training.h5py', 'r')

n_samples = len(f['train']['img']   )

for i in range(n_samples):
    img = f['train']['img'][i]
    kp_2D = f['train']['kp_2D'][i]
    img = img.transpose(1,2,0)
    plt.imshow(img)
    plt.plot(kp_2D[:,0], kp_2D[:,1])
    plt.show()
