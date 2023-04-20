# Model Parameter Identification via a Hyperparameter Optimization Scheme for Autonomous Racing Systems

## Abstract
In this letter, we propose a model parameter identification method via a hyperparameter optimization scheme (MI-HPO). Our method adopts an efficient explore-exploit strategy to identify the parameters of dynamic models in a data-driven optimization manner. We utilize our method for model parameter identification of the AV-21, a full-scaled autonomous race vehicle. We then incorporate the optimized parameters for the design of model-based planning and control systems of our platform. In experiments, MI-HPO exhibits more than 13 times faster convergence than traditional parameter identification methods. Furthermore, the parametric models learned via MI-HPO demonstrate good fitness to the given datasets and show generalization ability in unseen dynamic scenarios. We further conduct extensive field tests to validate our model-based system, demonstrating stable obstacle avoidance and high-speed driving up to 217 km/h at the Indianapolis Motor Speedway and Las Vegas Motor Speedway. The source code for our work and videos of the tests are available at https://github.com/hynkis/MI-HPO.

## Comments
- Preprint paper: https://arxiv.org/abs/2301.01470
- Submitted to [IEEE Control System Letters (L-CSS)](https://ieeexplore.ieee.org/abstract/document/10102102) (Accepted)

## Video demonstration
- Field tests at the Indianapolis Motor Speedway (IMS) and Las Vegas Motor Speedway (LVMS)
[![Youtube video](http://img.youtube.com/vi/A95ZCIqpmJw/0.jpg)](https://youtu.be/A95ZCIqpmJw)

## Usage
```bash
# Run model identification. A result would be saved as '.csv' in results directory.
python3 run_model_identification.py

# Plot the learned model with data.
python3 plot_learned_model.py
```

## Results of Model Parameter Identification (Sample tire dataset)
Since the dataset of the race vehicle is confidential, we created a random (but reasonable) sample tire dataset to run our codes.
<p align="center">
    <img src = "results/result_FyA_FyB.png" width="60%" height="60%">
</p>

## References
Please consider to cite this paper in your publications if this repo helps your research: <https://arxiv.org/abs/2301.01470>

    @article{seong2023model,
      title={Model Parameter Identification via a Hyperparameter Optimization Scheme for Autonomous Racing Systems},
      author={Seong, Hyunki and Chung, Chanyoung and Shim, David Hyunchul},
      journal={IEEE Control Systems Letters},
      year={2023},
      publisher={IEEE}
    }
    
We refer the following for Hyperband:
- Hyperband paper: <https://arxiv.org/abs/1603.06560>
- Implementation based on [hyperband](https://github.com/bkj/hyperband)

## Disclaimer

For any question, please contact [Hyunki Seong](https://github.com/hynkis).
