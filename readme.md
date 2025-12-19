# LFI meets Image-Transformation Model Inversion

This project develops an image-oriented framework for performing Likelihood-Free Inference (LFI) on parameterized image transformation simulators. The goal is to evaluate whether modern LFI algorithms can recover semantic parameters when the observations correspond to images produced by stochastic geometric deformations (e.g., Free-Form Deformations, FFD). LFI algorithms aim to estimate the posterior $\pi(\mathbf{\theta}|\mathbf{x}_o)$ from observed data $\mathbf{x}_o \sim p(\mathbf{x}|\mathbf{\theta})$. Where the likelihood $p(\mathbf{x}_o|\mathbf{\theta})$ is not available neither computational tractable, but samples can be simulated from an arbitrary $\mathbf{\theta}$.  

$$
\begin{gathered}
\underbrace{\pi(\mathbf{\theta}|\mathbf{x}_o)}_{\text{goal}} \propto \underbrace{p(\mathbf{x}_o|\mathbf{\theta})}_{\text{intractable}}\pi(\mathbf{\theta})\quad \quad \text{Bayesian posterior on high-dimensional data. Impractical with NDE.}\\
\underbrace{\pi(\mathbf{\theta}|s(\mathbf{x}_o))}_{\text{goal}} \propto \underbrace{p(s(\mathbf{x}_o)|\mathbf{\theta})}_{\text{intractable}}\pi(\mathbf{\theta})\quad \quad \text{Bayesian posterior on low-dimensional statistics. Possible with NDE.}\\
\end{gathered}
$$

 According to the literature, the contemporary approach to do this is by carrying out several sampling/simulation rounds, working together with a neural density estimator (NDE) network. Sampling is performed by establishing a hypothetical prior $\theta\sim\pi(\theta)$. Whereas, simulation requires defining an stochastic process parameterized by $\theta$. Previous works have shown that LFI can be performed accurately and efficiently on simple low-dimensional parameterized models ($\mathbb{x}\sim p(x|\mathbb{\theta})$ is parameterized by $\theta$). On the other hand, recent LFI algorithms implementations have showcased that the use of learned near-sufficient statistics yield to more accurate posterior estimation. Nevertheless, simple MLP models are being employed for these approximate sufficient statistic.

To the best of our knowledge, there are no current works employing parameterized image transformations as the simulators, such that we can test SOTA LFI algorithms on raw images. The final goal of this work is to verify whether current LFI methods can recover relevant information about semantic parameters $\mathbf{\theta}$ used to generate the deformed images. Accordingly, we are following the work of [Chen et al. (ICLR 2021)](https://openreview.net/pdf?id=SRDuJssQud), on using near-sufficient statistics (trained according to the [InfoMax Principle](https://www.microsoft.com/en-us/research/blog/deep-infomax-learning-good-representations-through-mutual-information-maximization/)) to simplify LFI. In summary, our main contribution is to replace the MLP networks, adopted in previous works, by CNNs, such that LFI algorithms can be tested with images as the observed data. The proposed framework is shown below:      


<img src=images/proposed_framework.jpg width=700>

The diagram of the work by [Chen et al. (ICLR 2021)](https://openreview.net/pdf?id=SRDuJssQud) is shown below:

<img src=images/original.svg width=700>

**Contributors**: Paulo L., Chevady C. Laguna D.

----------------------------------
### Proposed experiments
|Experiments|Purpose|Description|
|-----------|-------|-------|
|FFD parameter scaling| Since the variance used to generate the deformation parameters $\mathbb{\mu}$ is very small (warping is not easily distinguishable on the images) we need to scale up these parameters. |Fix the MI estimator to JSD, fix the NDE to MAF, Switch between scales $K_{\text{FFD}}\in\{1.0,3.0,4.0,5.0\}$.|
| Ablations on different MI estimators| The chosen MI estimator is critical for training the statistic network, we have to perform ablations on these estimators| Fix the architecture (StatNet,NDE), scale $K=3.0$, and switch the MI estimator between JSD (Jensen-Shannon Divergence), DC (Distance Correlation), DV (Donsker-Varadhan), WD (Wasserstein Distance).|
| Different NDEs (Neural Density Estimator)| Verify the impact on using different NDE models|Fix the MI estimator to JSD, fix the scale $K_{\text{FFD}}=4.0$, switch between NDE models (MAF or NDN). Notably, the implementation of MDN is still experimental.| 
---------------------------------
### FFD image warping process

<img src=images/original.jpeg width=200>
<img src=images/bsplines.jpeg width=200>
<img src=images/warped.jpeg width=200>
----------------------------------

### Repository organization
 
 **/examples**: include the _pynotebooks_ for running experiments on the proposed simulators (_FFDParam_, _FFDImage_)

 **/images_test**: source images used to generate warped images.

 **/lfi/algorithms/** : Main LFI algorithms (only SNL+ has been currently implemented)

 **/lfi/detsimul/** : Main problems (sample/simulation setups). Ours correspond to _problem_FFD.py_  and _problem_FFD_image.py_ 

**/lfi/neuralde/** : _MAF.py_ contains the implementation of Masked Autorregresive Flow. _MDE.py_ is our experimental implementation.

**/lfi/statnet/** : Implementation of the networks shown in the proposed framework illustration.

**/lfi/utils/** : Quantitative metrics (_discrepancy.py_), distribution objects (_distributions.py_), optimization for training (_optimizer.py_), functions for distribution and sample plotting (_visualization.py_).

## Getting started

The main scripts for running the proposed experiments are included in the examples folder:

- **/examples/task_FFD.ipynb**: E²SNL+ on _FFDParam_
- **/examples/task_FFD_image.ipynb**: E²SNL+ on _FFDImage_

