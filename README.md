# Decentralized Text-Based Person Re-Identification in Multi-Camera Networks
This is the code repository for the paper titled **Decentralized Text-Based Person Re-Identification in Multi-Camera Networks**. The repository contains the implementation of U-TextReIDNet and its decentralized deployment in multi-camera networks. The prototype system addresses limitations in scalability, bandwidth usage, and processing bottlenecks typically found in traditional centralized systems by enabling person re-ID directly at the camera node level.

<br>
<img src="docs/overall_system_architecture.png" width="50%">
The architecture of the proposed decentralized system.

<br><br>
<img src="docs/results_demo_video.png" width="100%"><br>
The screenshot of the prototype system (user application).


&nbsp;
## Requirements (dependencies)
### For Computer (Application Server) 
- Operating System: Ubuntu 20.04.6 LTS (irrelevant but worth mentioning)
- CUDA Version: 12.4
- python version: 3.8.10
- pytorch version: 1.13.1+cu117
- torchvision version: 0.14.1+cu117
- pillow version: 9.5.0
- opencv-python version: 4.8.1.78
- tqdm version: 4.66.1
- numpy version: 1.26.2
- natsort version: 8.4.0
