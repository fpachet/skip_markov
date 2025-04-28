# Experiment with skip markov models

Skip Markov models learn probabilities of events conditioned by events in the past located at some distance k.
The goal of this project is to study how to use them to make powerful predictions, especially compared to variable-order Markov models and max entropy models (Cf www.github.com/fpachet/max_entropy_music)


## Features

- a mixture of order-1 Markov skip model with parameter estimation using MLE
- the same, but adding predictions from future events (forward skips)
- 
## Authors
- [François Pachet](https://github.com/fpachet)

### Dependencies

The project requires the following Python packages:
numpy~=2.2.3
mido~=1.2.10
scipy==1.15.2



