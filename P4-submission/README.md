#Machine Learning Nanodegree Project-04:Train a Smart cab to drive.
##Sanjiv Lobo,2016


###Task 1:Implement a basic driving agent

Mention what you see in the agent’s behavior. Does it eventually make it to the target location?

####Implementing the basic driving agent

The initial driving agent was created with a greedy approach. It does better than choosing random values. It uses a greedy strategy and picks a random action to execute. After execution, it checks the reward if it’s better than the earlier reward or not. If a bad reward comes, a random action is executed otherwise it performs an initiated action. 
Yes it does make it to the destination since the grid is a closed loop.
Since enforce deadline is false, we do get 100% accuracy but most of the time the model overshoots the deadline parameter giving the model very bad efficiency.

####Identifying the states

Here are some of the possible percepts that were candidates for selection:
    Deadline
    Next waypoint by the planner
    Traffic light
    Traffic states (oncoming, left, right etc.)

From all the inputs, the following state variables were chosen for this model:

    Next waypoint
    Traffic light
    Traffic states(oncoming)


Some of the ideal candidates in this case would be the destination or the location. But, here are some of the issues that could occur:

    Destination : It changes all the time, therefore making it a state would be useless as the agent would not learn properly.
    Location : The sheer size of the grid would cause problems, since for the q values to converge, it would need  more trials.

Also, in order to model the destination as a part of the state, next waypoint was used.
  
Choosing traffic lights helps in modelling the traffic rule and also helps in training the cab to perform legal moves.

Choosing traffic state like oncoming helps in following traffic rules and making the model more realistic.


###Task 2: Working of the Q-learning agent
States used are traffic light, oncoming and next waypoint.
The best action function determines the best action by checking the best possible q value,
The policy function gets  the best q-value for any given state. Once the action is performed, we get the reward which updates the q-values. After few runs, the true q-values for each state are found, which helps in training the model  and after every run the q table is updated,

####Test Results for the Q-learning agent
This is the initial configuration for the Q-learning agent( parameters alpha, gamma and beta)

    Initial Q-value was set to a hypothetical value of 10. Initially, random decisions are made.
    Alpha value(learning rate) was set to 0.6, this value was found by trial and error over many values for alpha.
    Gamma value(discount value) was set to 0.4, set similarly by trial and error..

The results improved a lot and an accuracy of more than 93% was achieved with enforce deadline is True, This is a great improvement compared to the basic agent and its accuracy within a deadline.
Also, the cab reaches the destination with a positive large cumulative reward.
Thus, we can say that our agent has learnt an optimal policy by taking the shortest legal route.
