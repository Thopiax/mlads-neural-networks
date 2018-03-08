import subprocess
import os
from time import time

def submit_condor_job(i, t, name, low, high, samples):
    print("Submitting jobs for {name}={low} to {name}={high}, samples={samples}"
          .format(name=name, low=low, high=high, samples=samples))

    filename = '{name}_{low}_{high}'.format(low=low,
                                            high=high,
                                            name=name)
    filename = 'temp.cmd'

    with open(filename, 'w') as f:
        f.write(
         'universe = vanilla \n'
         'InitialDir =  \n'
         'executable = condor/condorSetupRun.sh  \n'
         'input = /dev/null \n'
         'output = condor/results/{name}.{i}.out \n'
         'error = condor/results/{name}.{i}.err \n'
         'log = condor/results/{name}.{i}.log \n'
         'arguments = {name} {t} {low} {high} {samples} false\n'
         'queue 1'.format(i=i, t=t, name=name, low=low, high=high, samples=samples))

    subprocess.call('condor_submit {filename}'.format(filename=filename), shell=True)

    os.remove(filename)


def main():
    t = time()
    for i in range(0, 100):
        submit_condor_job(i, t,  'hidden_layer_neurons',
                          low=1,
                          high=4,
                          samples=8)

if __name__ == "__main__":
    main()
