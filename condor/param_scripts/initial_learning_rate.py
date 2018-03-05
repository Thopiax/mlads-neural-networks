import subprocess


def submit_condor_job(name, low, high, samples):
    print("Submitting jobs for {name}={low} to {name}={high}, samples={samples}"
          .format(name=name, low=low, high=high, samples=samples))

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
         'arguments = {name} {low} {high} {samples}\n'
         'queue 1'.format(name=name, low=low, high=high, samples=samples))

    subprocess.call('condor_submit {filename}'.format(filename=filename), shell=True)


def main():
    for i in range(0, 2):
        submit_condor_job('lr',
                          low=i * 0.005,
                          high=i * 0.005 + 0.005,
                          samples=5)


if __name__ == "__main__":
    main()
