# Data-Driven Model Identification via Hyperparameter Optimization for Autonomous Racing Systems

## Abstract
In this letter, we propose a model identification method via hyperparameter optimization (MIHO). Our method adopts an efficient explore-exploit strategy to identify the parameters of dynamic models in a data-driven optimization manner. We utilize MIHO for model parameter identification of the AV-21, a full-scaled autonomous race vehicle. We then incorporate the optimized parameters for the design of model-based planning and control systems of our platform. In experiments, the learned parametric models demonstrate good fitness to given datasets and show generalization ability in unseen dynamic scenarios. We further conduct extensive field tests to validate our model-based system. The tests show that our race systems leverage the learned model dynamics and successfully perform obstacle avoidance and high-speed driving over $200 km/h$ at the Indianapolis Motor Speedway and Las Vegas Motor Speedway. The source code for MIHO and videos of the tests are available at https://github.com/hynkis/MIHO.

## Comments
- Preprint paper will be available soon.
- Submitted to [IEEE Control System Letters (L-CSS)](http://ieee-cssletters.dei.unipd.it/)

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

## Results of Model Identification (Sample tire dataset)
Since the dataset of the race vehicle is confidential, we created a random (but reasonable) sample tire dataset to run our codes.
<p align="center">
    <img src = "results/result_FyA_FyB.png" width="60%" height="60%">
</p>

## References
Please consider to cite this paper in your publications if this repo helps your research: <https://arxiv.org/abs/2301.01470>

    @article{seong23MIHO,
      title={Data-Driven Model Identification via Hyperparameter Optimization for the Autonomous Racing System},
      author={Seong, Hyunki and Chung, Chanyoung and Shim, David Hyunchul},
      journal={arXiv preprint arXiv:2301.01470},
      year={2023}
    }
    
We refer the following for Hyperband:
- Hyperband paper: <https://arxiv.org/abs/1603.06560>
- Implementation based on [hyperband](https://github.com/bkj/hyperband)

## Disclaimer

For any question, please contact [Hyunki Seong](https://github.com/hynkis).
