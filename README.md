My code for the MEDIQA-OE challenge.

I use a sophisticated approach which is described in this figure:

<img width="1600" height="1203" alt="image" src="https://github.com/user-attachments/assets/dc892ce9-a9fd-4482-94b8-1efc389c207d" />

This repo features:

- A notebook I used for playing around with the dev set
- A notebook I used for predicting on the test set (+ a version running a 32B model with a fallback to the 14B model's predictions if context size is exceeded!)
- An ugly hack of the original evaluation code to call it on a sample level / get a dict of metrics returned, so I can stick to my professional all-jupyter-workflow.

PS: I swear I usually write better code. Said better code just goes to a different school, in Canada. But it's real.
