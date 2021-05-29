<h1 align="center">FARLE: Framework for Atari Reinforcement Learning Environment</h1>

<div align="center">
<img src = "https://user-images.githubusercontent.com/33197180/111116948-cd3a5580-858c-11eb-92d6-12aec493b91f.png" width="500" />

<br>
<img src = "https://img.shields.io/badge/Made_with-Python3-blue?style=for-the-badge&logo=python" />
<img src = "https://img.shields.io/badge/Made_with-PyTorch-blue?style=for-the-badge&logo=pytorch"/>
<img src = "https://img.shields.io/badge/Made_with-OpenAI_Gym-blue?style=for-the-badge&logo=openai" />

<br>

</div>

---

### About:

**Framework for Atari Reinforcement Learning Environment (FARLE)** is a reinforcement learning framework made with PyTorch, built on top of OpenAI Gym to allow training of any Atari game from the ALE environment and perform transfer learning across games. The core algorithm used for performing training of Atari Games in this framework is Deep Q-Networks (DQNs) as of now.

### Problems it solves:

* A lot of research progress has been made in the field of reinforcement learning for training video games. However, a generic framework or template to easily train games was not easily available.
* A transfer learning framework for RL-based video game training is not found conveniently.
* Lack of easy-to-use tools to train models using OpenAI Gym Wrapper.
* This framework allows users to play with a lot of different parameters, giving users full liberty to customise their models as they wish.
* It allows people will less backgrund in Reinforcement Learning get started with building models without having to understand each and every technicality.
* It provides for robust experimentation as it maintains configuration and logs of model training, helping users analyze how much their model took time to train, their its performance, rewards achieved over time, improvements seen with transfer learning, etc. 

---

### Commands to use in FARL

1. Training a model from scratch:
	
```console
$ python3 main.py --name "Breakout-Scratch" \
                --env "BreakoutNoFrameskip-v4" \
                --replay_memory_size 50000 \
                --replay_start_size 25000 \
                --episodes 50000 \
		-- cuda
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
		-- cuda
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
		-- cuda
```

---

### Features:

* CLI-based parameter settings for experimentation.
* Supports both CPU and GPU.
* Model checkpointing for both scratch training and transfer learning, so that one can train models with limited GPU access by doing training in different time batches.
* Logging system for plotting different metrics and analyze the experiments performed. 


### Future scope of this project:

* Implement other algorithms like A3C, etc. for training and transfer learning of Atari games. 
* Add TensorBoard support.
* Add support for recording gameplay for specific training periods.
* Add TPU support.

### To run the project:

* [Fork](https://github.com/RL-LY-Project/Atari-Transfer-Learning) this Repository.
* cd into the directory in the terminal and run as:
  -`pip install -r requirements.txt`
* Run the commands above according to your needs.


#### This project still has scope of development, so you can also contribute to this Project as follows:
* [Fork](https://github.com/RL-LY-Project/Atari-Transfer-Learning) this Repository.
* Clone your Fork on a different branch:
	* `git clone -b <name-of-branch> https://github.com/RL-LY-Project/Atari-Transfer-Learning.git`
* After adding any feature:
	* Go to your fork and create a pull request.
	* We will test your modifications and merge changes.

---

<h3 align="center"><b>Developed with ❤️ by: </b></h3>
<div align="center">
<table style="border:1px solid black;margin-left:auto;margin-right:auto;">  
  <tr>
<td>
  <img algin ="center" src="https://avatars3.githubusercontent.com/u/33197180?s=150&v=4"/>
     
    Arghyadeep Das

<p align="center">
<a href = "https://github.com/arghyadeep99"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/arghyadeep-das/"><img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/></a>
</p>
</td>

<td>
  <img align='center' src="https://user-images.githubusercontent.com/33197180/114586227-0f48db00-9ca2-11eb-8211-aeb8a16440e3.jpeg" width="150" height="150">
     
    Vedant Shroff

<p align="center">
<a href = "https://github.com/vedant-shroff"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/vedant-shroff-31015615a/"><img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/></a>
</p>

<td>
  <img align='center' src="https://media-exp1.licdn.com/dms/image/C4D03AQFG6myoYUcwOw/profile-displayphoto-shrink_800_800/0/1617183828622?e=1623888000&v=beta&t=XIVx-0VISyhJFPSN8o2Txieink0lxb_Tu9rxrTRlZwI" width="150">
     
     Avi Jain

<p align="center">
<a href = "https://github.com/aviiiij"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/aviiii/"><img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/></a>
</p>
</td>
