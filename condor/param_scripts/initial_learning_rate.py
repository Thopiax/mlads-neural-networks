import subprocess
import numpy as np
from ../../


def test_param_values(name, low, high, samples):
    for value in np.linspace(low, high, num=samples, endpoint=False):
        print("Testing param values for {}={}".format(name, value))

        filename = '{name}_{low}_{high}'.format(low=low,
                                                high=high,
                                                name=name)
        filename = 'temp.cmd'

        with open(filename, 'w') as f:
            f.write(
             'universe = vanilla \n'
             'executable = /vol/bitbucket/vch15/mlads-neural-networks/condor/condorSetupRun.sh  \n'
             'output = condor/results/{name}.$(Process).out \n'
             'input = /dev/null \n'
             'error = condor/results/{name}.$(Process).err \n'
             'log = condor/results/{name}.log \n'
             'InitialDir =  /vol/bitbucket/vch15/mlads-neural-networks/ \n'
             'arguments = --{name} {value}\n'
             'queue 1'.format(name=name, value=value))

        subprocess.call('condor_submit {filename}'.format(filename=filename), shell=True)


def main():
    for i in range(0, 100):
        test_param_values('lr',
                          low=i * 0.005,
                          high=i * 0.005 + 0.005,
                          samples=5)


if __name__ == "__main__":
    main()
