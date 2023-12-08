# CSCE-642: Final Project

## PPO-Penalty
`python run.py -s ppo -t 3000 -d CartPole-v1 -e 3000 -a 0.001 -g 0.95 -l [32,32]`
## PPO-Clip
`python run.py -s ppo2 -t 3000 -d CartPole-v1 -e 3000 -a 0.001 -g 0.95 -l [32,32]`
## A2C
`python run.py -s a2c -t 3000 -d CartPole-v1 -e 3000 -a 0.001 -g 0.95 -l [32,32]`
## A3C
`python run.py -s a3c -t 3000 -d CartPole-v1 -e 3000 -a 0.001 -g 0.95 -l [32,32]`
## DQN
`python run.py -s dudqn -t 3000 -d CartPole-v1 -e 150 -a 0.01 -g 0.95 -c 0.95 -N 100 -b 32 -l [32,32]`
## DuDQN
`python run.py -s dqn -t 3000 -d CartPole-v1 -e 150 -a 0.01 -g 0.95 -c 0.95 -N 100 -b 32 -l [32,32]`
