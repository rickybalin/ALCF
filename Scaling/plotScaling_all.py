from matplotlib import pyplot as plt 
import numpy as np
import os

def main():
    # cases
    cases = {'v1': 256 ,
             'v2': 8, 
             'v3': 1024,
             'v4': 256,
             'v5': 256
            }

    
    plt.figure()
    # Loop over cases
    for key in cases:
        print(f'Dealing with case {key}\n')

        # Create list of scales to know which files to open
        nGPU = np.array([1],dtype=int)
        max_GPU = cases[key] 
        c = 0 
        while (nGPU[c]<max_GPU):
            tmp = nGPU[c]*2
            nGPU = np.append(nGPU,tmp)
            c += 1
        nScales = len(nGPU)

        # Loop over scales to collect throughput data
        tp = np.zeros(nScales)
        std = np.zeros(nScales)
        for i in range(nScales):
            fname = f'data_{key}_{nGPU[i]}.dat'
            if key!='v1': fname = f'{key}/'+fname
            if (os.stat(fname).st_size != 0):
                data = np.loadtxt(fname)
                tp[i] = np.mean(data)
                std[i] = np.std(data)

        # Plot scaling
        eff = tp/(tp[0]*nGPU)
        std = std/(tp[0]*nGPU)
        #plt.plot(nGPU, eff, 'o', label='Forcing')
        plt.errorbar(nGPU, eff, yerr=std, fmt='o', label=key)


    plt.xscale('log')
    plt.ylabel('Scaling Efficiency')
    plt.xlabel('Number of GPU')
    plt.title('Anisotropic SGS Model Scaling with DDP-PT on Polaris')
    plt.ylim((0,1.1))
    plt.legend()
    plt.grid()
    plt.savefig('scaling_anisoSGS_DDP-PT_Polaris.png')
    plt.show()



if __name__ == '__main__':
    main()

