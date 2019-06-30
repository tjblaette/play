# Let's play snake!

### \* This is unfinished work in progress \*

The goal is to evenutally implement the Snake game in python, allowing 
for both user- and AI- controlled game modes.

---
---


### Current functionalities

##### Play a game of snake
Run the following, supplying your choice of game world dimensions:

```
python play.py  10  10
```

##### Train an AI to play
Run the following, again supplying your choice of game world dimensions:

```
python train.py  10  10
```

Several parameters are available to tune training, list these and 
their current defaults with:

```
python train.py  -h
```

For a quick test, for example, run:

```
python train.py \
        --epochs 3 \
        --trainings 3 \
        --batch 12 \
        --gamma 0.9 \
        --explore 0.1 \
        --prefix 01 \
        --suffix quick-test \
        10 \
        10
```

##### Watch an AI play
Once you've trained an AI, watch it play but supplying the folder
which contains the saved model:

```
python simulate.py  10  10  01_10_x_10_3_x_3_x_12_0.9_quick-test_1561886392
```
