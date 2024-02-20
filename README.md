# $z = f(x, y)$
$x: Treatment$  
$y: Covariates$  
$z: Outcome$  

&nbsp;&nbsp; $\text{causal effect of x on z}$  
$=\text{sensitivity of change of z with respect to x, with y held constant}$   
$= \text{partial derivative of z  with respect to x}$   
$= \frac{\partial f}{\partial x}=\lim _{h \rightarrow 0} \frac{f(x+h, y)-f(x, y)}{h}$   
# 
In this repo, simulated data, denoted as $z = f(x, y)$, will be fed into a super learner to train a model. The trained model will subsequently generate values of $z$ within a specified range of $x$ and $y$ using  _Monte Carlo method_. The expression $\frac{f(x + \Delta x, y) - f(x - \Delta x, y)}{2 \Delta x}$ will be used to compute the partial derivative of $z$ with respect to $x$.  
  
![partial_derivative_as_slope](https://github.com/XinyueJia/Data_sim/assets/77334755/511bf349-50a9-437a-a15d-2917c83184ad)
