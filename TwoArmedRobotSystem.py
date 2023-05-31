from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets


class TwoArmedRobotSystem():
    def __init__(self, lengthOfFirstArm, lengthOfSecondArm):
        # Trajectory is an array composed of two arrays
        self.yCoordinates=None
        self.xCoordinates = None
        self.lengthOfFirstArm = lengthOfFirstArm
        self.lengthOfSecondArm = lengthOfSecondArm
        self.p_x=0
        self.p_y=0

    def getResidualCostFunction(self, arrayTheta):
        """
        getResidualCostFunction Is the system that governs the residual vector being equal to zero.

        :param: arrayTheta : The list containing the values of theta1 and theta2
        :return: returns the aforementioned system


        """

        theta1, theta2 = arrayTheta[0], arrayTheta[1]
        return [self.lengthOfFirstArm * np.cos(theta1) + self.lengthOfSecondArm * np.cos(theta1 + theta2) - self.p_x,
                self.lengthOfFirstArm * np.sin(theta1) + self.lengthOfSecondArm * np.sin(theta1 + theta2) - self.p_y]

    def getResidualMagnitudeSquared(self, arrayTheta):
        """
        getResidualMagnitudeSquared Is the magnitude of the residual squared

        :param: arrayTheta : The list containing the values of theta1 and theta2
        :return: returns the aforementioned value

        """
        theta1, theta2 = arrayTheta[0], arrayTheta[1]
        return (self.lengthOfFirstArm * np.cos(theta1) + self.lengthOfSecondArm * np.cos(theta1 + theta2) - self.p_x) ** 2 + \
               (self.lengthOfFirstArm * np.sin(theta1) + self.lengthOfSecondArm * np.sin(theta1 + theta2) - self.p_y) ** 2

    def jacobianOfResidual(self, arrayTheta):
        """
        jacobianOfResidual Is the 2d vector containing the first order derivative of the magnitude of the residual squared

        :param: arrayTheta : The list containing the values of theta1 and theta2
        :return: returns the aforementioned value

        """
        theta1 = arrayTheta[0]
        theta2 = arrayTheta[1]

        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        c12 = np.cos(theta1 + theta2)
        s12 = np.sin(theta1 + theta2)

        A = self.lengthOfFirstArm * s1 + self.lengthOfSecondArm * s12
        B = self.lengthOfFirstArm * c1 + self.lengthOfSecondArm * c12

        return np.array([2 * ((-A) * (B - self.p_x) + B * (A - self.p_y)),
                         2 * self.lengthOfSecondArm * ((-s12) * (B - self.p_x) + c12 * (A - self.p_y))])

    def hessienneOfResidual(self, arrayTheta):
        """
        jacobianOfResidual Is the 2d vector containing the second order derivative of the magnitude of the residual squared

        :param: arrayTheta : The list containing the values of theta1 and theta2
        :return: returns the aforementioned value

        """

        theta1 = arrayTheta[0]
        theta2 = arrayTheta[1]

        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        c12 = np.cos(theta1 + theta2)
        s12 = np.sin(theta1 + theta2)

        A = self.lengthOfFirstArm * s1 + self.lengthOfSecondArm * s12
        B = self.lengthOfFirstArm * c1 + self.lengthOfSecondArm * c12

        return np.array([[2 * (self.p_x * B + self.p_y * A), 2 * self.lengthOfSecondArm * (self.p_x * s12 + self.p_y * c12)],
                         [2 * self.lengthOfSecondArm * (self.p_x * c12 + self.p_y * s12),
                          2 * self.lengthOfSecondArm * (
                                      (-c12) * (B -self.p_x) - s12 * (A - self.p_y) + self.lengthOfSecondArm)]])

    def ResidualMagnitudeSquared(self,theta1, theta2):
        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        c12 = np.cos(theta1 + theta2)
        s12 = np.sin(theta1 + theta2)

        A = self.lengthOfFirstArm * c1 + self.lengthOfSecondArm * c12
        B = self.lengthOfFirstArm * s1 + self.lengthOfSecondArm * s12

        return (A - self.p_x) ** 2 + (B - self.p_y) ** 2



    def SolutionsVisualisation(self, theta1, theta2):
        @interact(k=widgets.IntSlider(min=0, max=len(theta1) - 1, step=1))  # decorateur qui prends
        def f(k):
            firstReachedPointX = self.lengthOfFirstArm * np.cos(theta1[k])
            firstReachedPointY = self.lengthOfSecondArm * np.sin(theta1[k])

            secondReachedPointX = firstReachedPointX + self.lengthOfFirstArm * np.cos(theta1[k] + theta2[k])
            secondReachedPointY = firstReachedPointY + self.lengthOfSecondArm * np.sin(theta1[k] + theta2[k])

            # Input Position of (x0,y0)
            x0 = 0
            y0 = 0
            figure, axes = plt.subplots()

            # To plot the arm and not only the points
            firstPortionArmInitPoint = [x0, firstReachedPointX]
            firstPortionArmEndPoint = [y0, firstReachedPointY]

            secondPortionArmInitPoint = [firstReachedPointX, secondReachedPointX]
            secondPortionArmEndPoint = [firstReachedPointY, secondReachedPointY]

            plt.plot(firstPortionArmInitPoint, firstPortionArmEndPoint, 'r')
            plt.plot(secondPortionArmInitPoint, secondPortionArmEndPoint, 'b')

            desiredPointX = self.xCoordinates[k]
            desiredPointY = self.yCoordinates[k]

            plt.scatter(self.xCoordinates, self.yCoordinates, label='The desired Point', color='r')

            xStarting=self.xCoordinates[0]
            yStarting=self.yCoordinates[0]
            xGoal=self.xCoordinates[-1]
            yGoal = self.yCoordinates[-1]

            plt.scatter(xStarting, yStarting, label='start', color='y')
            plt.scatter(xGoal, yGoal, label='goal', color='g')

            x = np.linspace(-1, 1, 150)
            y = np.linspace(-1, 1, 150)

            smallestUnReachableRegionRadius, biggestReachableRegionRadius=self.computeReachableRegion()

            a, b = np.meshgrid(x, y)
            # A circle whose center is the point of origin
            outerReachableRegion = a ** 2 + b ** 2 - biggestReachableRegionRadius
            innerUnReachableRegion = a ** 2 + b ** 2 - smallestUnReachableRegionRadius

            axes.contour(a, b, outerReachableRegion, [0])
            axes.contour(a, b, innerUnReachableRegion, [0])
            axes.set_aspect(1)

            plt.title('Center-Radius form Circle')
            plt.legend()
            plt.grid()
            plt.show()

    def computeReachableRegion(self):
        """
         computeReachableRegion : will compute the radii of the inner and outer reachable region

        :returns:
            smallestUnReachableRegionRadius : The radius of the inner unreachable region
            biggestReachableRegionRadius : The radius of the outer reachable region

        """
        smallestUnReachableRegionRadius = (self.lengthOfFirstArm - self.lengthOfSecondArm) ** 2
        biggestReachableRegionRadius = (self.lengthOfFirstArm + self.lengthOfSecondArm) ** 2

        return smallestUnReachableRegionRadius, biggestReachableRegionRadius

    def computeTrajectory(self):
        """
         computeTrajectory : will compute the desired trajectory of the robot


        """
        print("Greetings ! ")
        xStarting = float(input("Please enter the value x coordinate of the starting point\n"))
        yStarting = float(input("Please enter the value y coordinate of the starting point\n"))

        xGoal = float(input("Please enter the value x coordinate of the ending point\n"))
        yGoal = float(input("Please enter the value y coordinate of the ending point\n"))

        print("thank you")

        numberOfSamples = int(input("please enter the number of samples\n"))

        print("gratitude")
        sampleStepY = (yGoal - yStarting) / numberOfSamples
        sampleStepX = (xGoal - xStarting) / numberOfSamples

        self.xCoordinates = [(xStarting + sampleStepX * i) for i in range(numberOfSamples + 1)]
        self.yCoordinates = [(yStarting + sampleStepY * i) for i in range(numberOfSamples + 1)]

    def DisplayIsoValues(self,theDesiredPointIndex,theta1Min,theta2Min,nIso):

        # Displaying the isovalues taking into account one minimum

        p_x = self.xCoordinates[theDesiredPointIndex]
        p_y = self.yCoordinates[theDesiredPointIndex]

        thetaMin, thetaMax, nx = - 180, 180, 10
        oneDTheta1 = np.linspace(thetaMin, thetaMax, nx)
        oneDTheta2 = np.linspace(thetaMin, thetaMax, nx)
        twoDTheta1, twoDTheta2 = np.meshgrid(oneDTheta1, oneDTheta2)


        print(theta1Min)
        print(theta2Min)
        plt.contour(twoDTheta1, twoDTheta2, self.ResidualMagnitudeSquared(twoDTheta1, twoDTheta2), nIso)
        plt.scatter(theta1Min, theta2Min, label='M1', color='red')
        plt.plot()
        plt.title('Isovaleurs')
        plt.xlabel('Valeurs de Theta1')
        plt.ylabel('Valeurs de Theta2')
        plt.grid()
        plt.axis('square')





