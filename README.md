# Gaze Estimation and Demographic Classification System

## Abstract
Abstract
In recent years, more and more businesses are starting to deploy AI-based technologies to take advantage of the increasingly high amount of information that can be collected or is already available. Our system aims to help commercial activities such as retail shops to better understand the needs and preferences of their customers based on what actually captures customers' attention, thus improving commercial strategy development.

## Introduction
The problem of commercial strategy optimization has become central in recent years. We developed a solution to help shops better understand their customers' trends and preferences in order to better create commercial strategies. To do this we are proposing a Computer Vision based gaze estimation system paired with a demographic estimator, which aims to provide meaningful information about the products that capture the customer's attention the most. Our system is capable of understanding where the customer is looking and creating a heatmap, divided by category of customer, using which the shop would be able to better design the exposition strategy.
In the scenario we’ve thought about, there will be a fixed camera in shelves or shop windows, in order to have a fixed frontal image of a subject looking at the products. It is important that the camera has a fixed position: this allows the shop manager to correlate the heatmap of gazes created by the system with the position of the items in the shelf/window.
It’s also important to mention that in our case an extremely accurate gaze tracking is not necessarily a heavy constraint. This fact allows us to reduce the camera frame rate in order to simplify the problem and lighten the computational effort required.

## The architecture
The system is made up mainly by three components:
- A CNN based network for **gaze detection**.
- A CNN based network for **demographic classification** (thus gener and age estiamtion).
- A **retrieval component**, which helps the second network to improve its accuracy.

Following, a scheme that shows how hte three components interact together.

![image](https://github.com/AGMLProjects/GazeDetector/assets/37586010/ea9414b3-c1d6-484d-8a59-7bafd13c65cd)

## Demo

![demo1](https://github.com/AGMLProjects/GazeDetector/assets/37586010/4a2a818f-1b65-4d9b-a67a-f72025438c69)

## Bibliography

[1] Appearance-Based Gaze Estimation in the Wild - Zhang et al, Max Planck Institute for Informatics, Saarbrucken, Germany <br>
[2] It’s Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation - Zhang et al, Max Planck Institute for Informatics, Saarbrucken, Germany <br>
[3] Few-Shot Adaptive Gaze Estimation - Seonwook Park et al, ETH Zurich, NVIDIA <br>
[4] facenet-pytorch: https://github.com/timesler/facenet-pytorch <br>
[5] GRA_Net: A Deep Learning Model for Classification of Age and Gender From Facial Images - Avishek Grain, Biswarup Ray, Pawan Kumar Singh et al - Jadavpur University, The National University of Malaysia, IEEE <br>
[6] Deep Residual Learning for Image Recognition - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - IEEE <br>
[7] Automatic age and gender classification using supervised appearance model - A. M. Bukar, H. Ugail, and D. Connah - J. Electron. Imag., vol. 25, no. 6, Aug. 2016, Art. no. 061605 <br>
[8] Age and gender estimation by using hybrid facial features - V. Karimi and A. Tashk - Proc. 20th Telecommun. Forum (TELFOR), Nov. 2012, pp. 17251728. <br>
