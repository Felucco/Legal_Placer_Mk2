# Core Folder Structure
- **agent**: folder containing definitions of agent, policy network and strategy handling (*Epsilon Greedy* with exp and linear functions)
- **encoder**: folder containing the definition and training process for the autoencoder structure. It contains also the main file used to generate input samples
- **env**: folder containing the environment definition in terms of state-action handling and rewards
- **model_trainer**: unused folder containing experiments regarding the capability of a network to predict the effects of a placement in a floorplan
- **rep_memory**: folder containing the definitions of the repla memory used to store experiences

All the other files are placed in the main folder and are so organized:
- **agent_trainer.py**: main class used to define the training process. It defines the training loop and contains all initializations
- **train_test.py**: "main" file, it is used to launch the actual training process and to observe the results

# Environment settings
The environment is created inside the **train_test.py** main file with a fixed canvas size and a variable number of macros. Macros are created so not to exceed a fixed portion of the floorplan area

# Policy hyperparameters
Policy network hyperparameters can be controlled by changing definitions inside the agent trainer class. Class constans are then used to properly create the policy network.
**Encoder trainability**: the encoder section of the network is loaded with pre-trained weights and the user can choose whether to allow for finer training or to freeze such layers. As for now **best results** are obtained by allowing re-training of the encoder