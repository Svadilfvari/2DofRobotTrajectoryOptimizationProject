# Two-Armed Robot Optimization Project

## The Problem

The goal of this project is to implement optimization methods to reduce the residual \( R(\theta) \) resulting from inaccuracies in the inverse geometric model of a two-joint robotic arm.

## Results

A GUI has been added to display the different solutions. This GUI is modular, allowing users to choose the size of the robot arm, the trajectory, and the number of solutions to be displayed.


![Solution Visualization](https://github.com/Svadilfvari/2DofRobotTrajectoryOptimizationProject/blob/main/2Dof_Robot_Arm_GUI.gif)
## The Code

The code consists of two main classes:

1. **TwoArmedRobotSystem**: Defines the parameters of the problem, the cost function, and other methods used for minimization.
2. **MinimaToolBox**: Contains the optimization methods.

Refer to the UML diagram below for a detailed class structure.

![UML Diagram](https://github.com/Svadilfvari/2DofRobotTrajectoryOptimizationProject/blob/main/2DofRobotOptimizationUML.png)

### Optimization Methods

#### First Method: Root Finding

The geometric model is governed by the system (I):
$$\[\begin{cases}l_1 \cos(\theta_1) + l_2 \cos(\theta_1 + \theta_2) = p_x \\ l_1 \sin(\theta_1) + l_2 \sin(\theta_1 + \theta_2) = p_y\end{cases}\]$$

The residual is a function of two input variables $(\theta_1\)$ and $(\theta_2\)$ and two outputs (II):

$$\begin{cases}
l_1 \cos(\theta_1) + l_2 \cos(\theta_1 + \theta_2) - p_x = 0 \\
l_1 \sin(\theta_1) + l_2 \sin(\theta_1 + \theta_2) - p_y = 0
\end{cases}
$$

Our objective is to implement optimization methods to find solutions to system (II), i.e., values of \(\theta_1\) and \(\theta_2\) that satisfy (II). This is defined in the file `System.py` where we use the `root` method from `scipy`.

#### Second Method: Optimization

We use the `optimize` method from `scipy` to find the minimum of the cost function, which in our case is the squared norm of the residual (cf. system II). This gives the following equation:

$$\[||R(\theta)||² = (l_1 \cos(\theta_1) + l_2 \cos(\theta_1 + \theta_2) - p_x)² + (l_1 \sin(\theta_1) + l_2 \sin(\theta_1 + \theta_2) - p_y)²\]$$

#### Third Method: Newton's Method

To apply Newton's method, we need to solve:

$$\[H[J(X_n)] * ΔX = - ∇[J(X_n)]\]$$

In our case, \( X_n \) corresponds to \(\theta\) and \( J \) corresponds to \( R \). Thus, we have:

$$\[\begin{bmatrix}H_{11} & H_{12} \\H_{21} & H_{22}\end{bmatrix}*\begin{bmatrix}Δθ_1 \\Δθ_2\end{bmatrix}= -\begin{bmatrix}g_1 \\g_2\end{bmatrix}\]$$

This results in the following system (III):

$$
\begin{cases}
H_{11} * Δθ_1 + H_{12} * Δθ_2 + g_1 = 0 \\
H_{21} * Δθ_1 + H_{22} * Δθ_2 + g_1 = 0
\end{cases}
$$

Here, the unknown is \( Δθ = (Δθ_1, Δθ_2) \). Therefore, we use the `root` method on this system to solve it.

#### Jacobian and Hessian Matrices

The Jacobian and Hessian matrices are written as follows:
$$
\[∇(||R(θ)||²) =
\begin{bmatrix}
2(-l_1 \sin θ_1 - l_2 \sin (θ_1 + θ_2))(l_1 \cos θ_1 + l_2 \cos (θ_1 + θ_2) - p_x) + 2(l_1 \cos θ_1 + l_2 \cos (θ_1 + θ_2) - p_y) \\
2l_2(-\sin (θ_1 + θ_2)(l_1 \cos θ_1 + l_2 \cos (θ_1 + θ_2) - p_x) + \cos (θ_1 + θ_2)(l_1 \sin θ_1 + l_2 \sin (θ_1 + θ_2) - p_y))
\end{bmatrix}\]
$$
$$
\[
H(||R(θ)||²) =
\begin{bmatrix}
2 p_x (l_1 \cos θ_1 + l_2 \cos (θ_1 + θ_2)) \\
2 p_y (l_1 \sin θ_1 + l_2 \sin (θ_1 + θ_2)) \\
-\cos (θ_1 + θ_2)(l_1 \cos θ_1 + l_2 \cos (θ_1 + θ_2) - p_x) \\
-\sin (θ_1 + θ_2)(l_1 \sin θ_1 + l_2 \sin (θ_1 + θ_2) - p_y) + 2 l_2 \\
2 p_x l_2 \sin (θ_1 + θ_2) + 2 p_y l_2 \cos (θ_1 + θ_2) \\
2 l_2 (\cos (θ_1 + θ_2) p_x + \sin (θ_1 + θ_2) p_y)
\end{bmatrix}
$$
