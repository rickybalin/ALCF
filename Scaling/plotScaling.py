from matplotlib import pyplot as plt
import numpy as np

def main():
    # Create list of scales to know which files to open
    nGPU = np.array([1],dtype=int)
    max_GPU = 128
    c = 0
    while (nGPU[c]<max_GPU):
        tmp = nGPU[c]*2
        nGPU = np.append(nGPU,tmp)
        c += 1
    nScales = len(nGPU)

    # Loop over scales to collect throughput data
    tp = np.empty(nScales)
    std = np.empty(nScales)
    for i in range(nScales):
         fname = 'data_'+str(nGPU[i])+'.dat'
         data = np.loadtxt(fname)
         tp[i] = np.mean(data[1:])
         std[i] = np.std(data[1:])

    # Plot scaling
    eff = tp/(tp[0]*nGPU)
    std = std/(tp[0]*nGPU)
    plt.figure()
    #plt.plot(nGPU, eff, 'o', label='Forcing')
    plt.errorbar(nGPU, eff, yerr=std, fmt='o', label='Forcing')
    plt.ylabel('Scaling Efficiency')
    plt.xlabel('Number of GPU')
    plt.title('ResNet18 Scaling with HVD-PT on ThetaGPU')
    plt.grid()
    plt.savefig('scaling_resnet18_HVD-PT_ThetaGPU.png')
    plt.show()
         


if __name__ == '__main__':
    main()
