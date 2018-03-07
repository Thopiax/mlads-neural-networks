import subprocess
import os


def submit_condor_job(i, name, low, high, samples):
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
         'arguments = {name} {low} {high} {samples}\n'
         'queue 1'.format(i=i, name=name, low=low, high=high, samples=samples))

    subprocess.call('condor_submit {filename}'.format(filename=filename), shell=True)

    os.remove(filename)


def main():
    for i in range(0, 50):
        submit_condor_job(i, 'lr',
                          low=i * 0.005,
                          high=i * 0.005 + 0.005,
                          samples=5)


if __name__ == "__main__":
    main()
