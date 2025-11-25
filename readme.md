## LFI meets DNN Model Inversion Attacks

This projects aims to develop DNNs based simulators for LFI (Likelihood-free inference) experiments. Previous works have shown that LFI can be performed accurately and efficiently (e.g. use of near-sufficient statistics) on parameterized models (the prior $x\sim \pi(x)$ is parameterized by $\theta$). Nevertheless, there are no works employing DNNs as the simulators, such that we can test LFI algorithms on them, and verify whether they can recover relevant information about the data distribution used to train the simulator.   

- lfi (all the algorithms require an statistic Net and Density estimator Net):
    * neuralde: Neural Density Estimators
    * statnet: Statistic Networks
    * algorithms: LFI algorithms (posterior solver)