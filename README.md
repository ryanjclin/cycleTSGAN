## How to run:
1. cd to cycleTSGAN
2. training: python/python3 main.py fault_id
3. inference: python/python3 inference.py fault_id
4. For visualizing the gen output, use visualize_gen_data.ipynb 

## requirements
1. create a new conda env with python=3.8
2. For "pywt: pip install PyWavelets
3. other package just follow default version


## Experiment Todo:

### Ryan: 
1. optimize model(gen and dis),
2. loss fn,
3. determine training data size
4. (Steven and John can do it too if you want.)

### Steven and John: 
1. read SFKGAN paper (especially, experiment part), and synthetic data evaluation method in this paper (https://amulyayadav.github.io/AI4SG2023/images/7.pdf)
2. read the code, get familiar with it. (you can rewrite code. right now the way I save checkpoint and generated data is kinda shit)
3. use the synthetic data in generated_data/fault_01 to build evaluation model/method right now. After we generate all type of faulty data, we do a complete experiment
4. you could modify these evaluation approach or come up with new one.


## Paper writing Todo: (not now)

### Ryan: write abstract, introduction, methodology (model part)
### Steven and John: methodology (evaluation method part)
