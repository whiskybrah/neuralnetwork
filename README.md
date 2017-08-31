# NN Using SGD
![nn](http://i.imgur.com/gGryoGu.png)

Calculations for giving inputs to the Network

Finding net input for hidden layer 1 or h_1:
net_h1= w_(1)× k_(1) + w_(2)× k_2 + b_1 × 1 
net_h1= 0.1× 0.1+ 0.2 × 0.1+0.1 ×1=0.13

Logistic function to get output:
output_h1=  1/(1+e^(-net_h1 ) )
output_h1=  1/(1+e^(-0.13) )=0.53245430638 

Finding net input for hidden layer 2 or h_2:
net_h2= w_(3 )× k_(1 )+ w_(4 )× k_2+ b_1  ×1 
net_h2= 0.1× 0.1+ 0.1 × 0.1+0.1 ×1=0.12

Logistic function to get output:
output_h2=  1/(1+e^(-net_h2 ) )
output_h2=  1/(1+e^(-0.12) )=0.52996405176 

Using the output from the hidden layer neurons, we can use them as inputs for output layer neurons.
Using the first output neuron j_1:
net_j1= w_(5 )× output_(h1 )+ w_(6 )× output_h2 + b_2  ×1
net_j1= 0.1× 0.52996405176 + 0.1×0.52996405176+0.1 ×1=0.20599281035 
output_j1=  1/(1+e^(-net_j1 ) )
output_j1=  1/(1+e^(-0.20599281035) )=0.5513168699 

Repeated for second output j_2:
net_j1= w_(7 )× output_(h1 )+ w_8× output_h2+ b_2  ×1
net_j1= 0.1× 0.52996405176 + 0.2×0.52996405176+0.1 ×1=0.25898921552
output_j2=  1/(1+e^(-net_j2 ) )
output_j2=  1/(1+e^(-0.25898921552) )=0.56438780237
Using the sqaured error function the error can be calculated, then by summing them the total error can be found.

E_total= ∑ 1/2 (target-output)^2

The target output for j_1 is 1, but the neural network output is 0.5513168699 
E_j1=  1/2 (target_j1-output_j1)^2
E_j1=  1/2 (1-0.5513168699)^2=0.10065827561 

The target output for j_2 is 0, but the neural network output is 0.56438780237 
E_j2=  1/2 (target_j2-output_j2)^2
E_j2=  1/2 (1-0.56438780237)^2=0.09487899336 

Total error
E_total= E_j1+E_(j2 )=0.10065827561+  0.09487899336=0.19553726897 
