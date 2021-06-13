<h1 align="center">FARLE: Framework for Atari Reinforcement Learning Environment</h1>

<div align="center">
<img src = "https://user-images.githubusercontent.com/33197180/120876562-0e139a80-c5cf-11eb-856e-47d2c05f4909.png" width="300" />

<br>
<img src = "https://img.shields.io/badge/Made_with-Python3-blue?style=for-the-badge&logo=python" />
<img src = "https://img.shields.io/badge/Made_with-PyTorch-blue?style=for-the-badge&logo=pytorch"/>
<img src = "https://img.shields.io/badge/Made_with-OpenAI_Gym-blue?style=for-the-badge&logo=openai" />

<br>

</div>

---

### About:

**Framework for Atari Reinforcement Learning Environment (FARLE)** is a reinforcement learning CLI-tool made with PyTorch, built on top of OpenAI Gym to allow training of any Atari game from the ALE environment and perform transfer learning across games. The core algorithm used for performing training of Atari Games in this framework is Deep Q-Networks (DQNs) as of now.

### Problems it solves:

* A lot of research progress has been made in the field of reinforcement learning for training video games. However, a generic framework or template to easily train games was not easily available.
* A transfer learning framework for RL-based video game training is not found conveniently.
* Lack of easy-to-use tools to train models using OpenAI Gym Wrapper.
* This framework allows users to play with a lot of different parameters, giving users full liberty to customise their models as they wish.
* It allows people will less background in Reinforcement Learning get started with building models without having to understand each and every technicality.
* It provides for robust experimentation as it maintains configuration and logs of model training, helping users analyze how much their model took time to train, their its performance, rewards achieved over time, improvements seen with transfer learning, etc. 

---

### Commands to use in FARLE

1. Training a model from scratch:
	
```console
$ python3 main.py --name "Breakout-Scratch" \
                --env "BreakoutNoFrameskip-v4" \
                --replay_memory_size 50000 \
                --replay_start_size 25000 \
                --episodes 50000 \
		--cuda
```

2. Training a model via transfer learning:
	
```console
$ python3 main.py --name "Pong-from-Breakout" \
                --env "PongNoFrameskip-v4" \
                --pretrained True \
                --pretrain_model "./logs/Breakout-Scratch-01/model.pt" \
                --pretrain_env "BreakoutNoFrameskip-v4" \
                --replay_memory_size 50000 \
                --replay_start_size 25000 \
		--cuda
```

3. Resume scratch training of model from checkpoint:
	
```console
$ python3 main.py --name "Breakout-Resume" \
                --env "BreakoutNoFrameskip-v4" \
                --resume_train True \
                --resume_train_path "./logs/Breakout-Scratch-01/model.pt" \
                --resume_train_env "BreakoutNoFrameskip-v4" \
                --replay_memory_size 50000 \
                --replay_start_size 25000 \
                --episodes 50000 \
		--cuda
```

4. Resume transfer learning of model from checkpoint:
	
```console
$ python3 main.py --name "Pong-from-Breakout-Resume" \
                --env "PongNoFrameskip-v4" \
                --resume_train True \
		--resume_transfer_train True \
                --resume_train_path "./logs/Pong-from-Breakout-01/model.pt" \
                --resume_train_env "PongNoFrameskip-v4" \
                --replay_memory_size 50000 \
                --replay_start_size 25000 \
                --episodes 50000 \
		--cuda
```

---

### Features:

* CLI-based parameter settings for experimentation.
* Supports both CPU and GPU.
* Model checkpointing for both scratch training and transfer learning, so that one can train models with limited GPU access by doing training in different time batches.
* Logging system for plotting different metrics and analyze the experiments performed.
* Beginner-friendly graph plotting notebooks for logs recorded.


### Future scope of this project:

* Implement other algorithms like A3C, Rainbow, etc. for training and transfer learning of Atari games. 
* Add support for more games like flash games, retro games, etc. through OpenAI Universe.
* Add TensorBoard support for enhanced logging.
* Add Weights and Biases integration for tracking training progress and visualization of weights as they change throughout the training process.
* Add video recording capability for rendering gameplay training strategies during training and testing.
* Add TPU support.
* Add GUI.

### To run the project:

* [Fork](https://github.com/arghyadeep99/FARLE) this Repository.
* `cd` into the directory in the terminal and run as:
	* `pip3 install -r requirements.txt`
* Run the commands above according to your needs.


#### This project still has scope of development, so you can also contribute to this Project as follows:
* [Fork](https://github.com/arghyadeep99/FARLE) this Repository.
* Clone your Fork on a different branch:
	* `git clone -b <name-of-branch> https://github.com/arghyadeep99/FARLE.git`
* After adding any feature:
	* Go to your fork and create a pull request.
	* We will test your modifications and merge changes.

---

<h3 align="center"><b>Developed with ❤️ by <a href="https://github.com/arghyadeep99">Arghyadeep Das</a> and <a href="https://github.com/vedant-shroff">Vedant Shroff</a>. Logo by <a href="https://github.com/aviiiij">Avi Jain</a>.</b></h3>
