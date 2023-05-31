from TwoArmedRobotSystem import *
from scipy import optimize

global p_x
global p_y


class MinimaToolBox():
    def __init__(self,TwoArmedRobotSystem,accuracy):
        self.TwoArmedRobotSystem=TwoArmedRobotSystem
        self.accuracy=accuracy

    def getMinimaViaRoot(self):
        """
         getMinimaViaMinimize : will compute the abscissa of the root of the system (that is residual error equal to zero)
        :returns:
            the closest (depending on the accuracy) values for theta1 and theta2 that minimize the residual error
        """

        rootMethodeThetaOneValues = []
        rootMethodeThetaTwoValues = []

        xCoordinates = self.TwoArmedRobotSystem.xCoordinates
        yCoordinates = self.TwoArmedRobotSystem.yCoordinates
        numberOfSamples = len(xCoordinates)

        for i in range(numberOfSamples):
            self.TwoArmedRobotSystem.p_x = xCoordinates[i]
            self.TwoArmedRobotSystem.p_y = yCoordinates[i]

            sol = optimize.root(self.TwoArmedRobotSystem.getResidualCostFunction, [0, 0])
            rootMethodeThetaOneValues.append(sol.x[0])
            rootMethodeThetaTwoValues.append(sol.x[1])
        return rootMethodeThetaOneValues,rootMethodeThetaTwoValues

    def getMinimaViaMinimize(self):
        """
         getMinimaViaMinimize : will compute the abscissa of the minima of a function using the method "minimize"
                                of the library "Scipy"

        :returns:
            the closest (depending on the accuracy) values for theta1 and theta2 that minimize the residual error
        """

        minMethodThetaOneValues = []
        minMethodThetaTwoValues = []

        xCoordinates = self.TwoArmedRobotSystem.xCoordinates
        yCoordinates = self.TwoArmedRobotSystem.yCoordinates
        numberOfSamples = len(xCoordinates)

        for i in range(numberOfSamples ):
            self.TwoArmedRobotSystem.p_x = xCoordinates[i]
            self.TwoArmedRobotSystem.p_y = yCoordinates[i]

            sol = optimize.minimize(self.TwoArmedRobotSystem.getResidualMagnitudeSquared, [0, 0])
            minMethodThetaOneValues.append(sol.x[0])
            minMethodThetaTwoValues.append(sol.x[1])

        return minMethodThetaOneValues,minMethodThetaTwoValues

    def deltaThetaSystem(self,thetaArray):
        """
        deltaThetaSystem : Is the required to be solved system for the newton method defined as
                           deltaX = solution of Hessinne * DeltaX + gradJ =0

        :param: arrayTheta : The list containing the values of theta1 and theta2
        :return: returns the aforementioned system


        """

        deltaTheta1 = thetaArray[0]
        deltaTheta2 = thetaArray[1]
        # The components of the Hessienne Matrix
        h11 = hessResidual[0][0]
        h12 = hessResidual[0][1]
        h21 = hessResidual[1][0]
        h22 = hessResidual[1][1]
        # The components of the jacobian vector
        g1 = gradResidual[0]
        g2 = gradResidual[1]

        return [h11 * deltaTheta1 + h12 * deltaTheta2 + g1, h21 * deltaTheta1 + h22 * deltaTheta2 + g2]

    def getMinimaViaNewtonForOneValue(self,initialValueArray, accuracy, maxIter):

        """
         getMinimaViaNewtonForOneValue : will compute the abscissa of the minima of a function using Newton's method,
                                         it is designed to compute the minima for one coordinate, it'll be used later on
                                         in the method for computing the minimal for the whole set of coordinates

        :param: initialValueArray: is the first guess for the root
        :param: accuracy: the highest tolerable value of error
        :returns:
            the closest (depending on the accuracy) values for theta1 and theta2 that minimize the residual error
            "i" The number of iterations until the final result

            "error" An error estimator in case the algorithm has gone wrong
        """
        # Initialization

        theNthValueOfThetaArray = initialValueArray
        # We set those variables as global ones so that we can retrieve their values during
        # the process of looking for the solutions of  "DeltaThetaSystem"

        global gradResidual
        global hessResidual
        deltaThetaArray = []
        dX = 1
        i = 0
        error = True
        while dX > accuracy and i < maxIter:
            i += 1
            gradResidual = self.TwoArmedRobotSystem.jacobianOfResidual(theNthValueOfThetaArray)
            hessResidual = self.TwoArmedRobotSystem.hessienneOfResidual(theNthValueOfThetaArray)

            # deltaX = solution de Hessinne * DeltaX =- gradJ
            sol = optimize.root(self.deltaThetaSystem, [0, 0])
            deltaThetaArray = sol.x

            theNthValueOfThetaArray = np.add(theNthValueOfThetaArray, deltaThetaArray)

            dX = np.linalg.norm(deltaThetaArray)

            error = False

        return theNthValueOfThetaArray, i, error

    def getMinimaViaNewtonForMultipleValues(self,initialValueArray, accuracy, maxIter):

        """
         getMinimaViaNewtonForMultipleValues : will behave as the methode "getMinimaViaNewtonForOneValue"
                                               For the only exception that it'll (the latter) be used
                                               in a loop to compute the minimal for all the points

        :param: initialValueArray: is the first guess for the root
        :param: accuracy: the highest tolerable value of error
        :returns:
            the closest (depending on the accuracy) values for theta1 and theta2 that minimize the residual error
            "i" The number of iterations until the final result

            "error" An error estimator in case the algorithm has gone wrong
        """
        newtonMethodThetaOneValues = []
        newtonMethodThetaTwoValues = []

        xCoordinates = self.TwoArmedRobotSystem.xCoordinates
        yCoordinates = self.TwoArmedRobotSystem.yCoordinates
        numberOfSamples = len(xCoordinates)

        for j in range(numberOfSamples):
            self.TwoArmedRobotSystem.p_x = xCoordinates[j]
            self.TwoArmedRobotSystem.p_y = yCoordinates[j]

            theNthValueOfThetaArray, i, error = self.getMinimaViaNewtonForOneValue(initialValueArray, accuracy, maxIter)

            newtonMethodThetaOneValues.append(theNthValueOfThetaArray[0])
            newtonMethodThetaTwoValues.append(theNthValueOfThetaArray[1])

        return newtonMethodThetaOneValues,newtonMethodThetaTwoValues, i, error

    def getMinimaViaFixedStepGradForOneValue(self,initialValueArray, accuracy, maxIter, alpha):
        """
         getMinimaViaFixedStepGradForOneValue : will compute the abscissa of the minima of a function using Fixed step
                                                Gradient Method it is designed to compute the minima for one coordinate,
                                                it'll be used later on in the method for computing the minimal for the
                                                whole set of coordinates

        :param: initialValueArray: is the first guess for the root
        :param: accuracy: the highest tolerable value of error
        :returns:
            the closest (depending on the accuracy) values for theta1 and theta2 that minimize the residual error
            "i" The number of iterations until the final result

            "error" An error estimator in case the algorithm has gone wrong
        """
        # Initialization

        theNthValueOfThetaArray = initialValueArray
        # We set those variables as global ones so that we can retrieve their values during
        # the process of looking for the solutions of  "DeltaThetaSystem"

        theNextValueOfThetaArray = []
        dX = 1
        i = 0
        error = True

        global gradResidual
        while dX > accuracy and i < maxIter:
            i += 1
            gradResidual = self.TwoArmedRobotSystem.jacobianOfResidual(theNthValueOfThetaArray)

            alphaTimesGradResidual = alpha * gradResidual

            theNextValueOfThetaArray = np.subtract(theNthValueOfThetaArray, alphaTimesGradResidual)

            deltaThetaArray = np.subtract(theNextValueOfThetaArray, theNthValueOfThetaArray)

            dX = np.linalg.norm(deltaThetaArray)

            theNthValueOfThetaArray = theNextValueOfThetaArray

            error = False

        return theNthValueOfThetaArray, i, error

    def getMinimaViaFixedStepGradForMultipleValues(self, initialValueArray, accuracy, maxIter, alpha):

        """
         getMinimaViaFixedStepGradForMultipleValues : will behave as the methode "getMinimaViaFixedStepGradForOneValue"
                                                      For the only exception that it'll (the latter) be used
                                                      in a loop to compute the minimal for all the points

        :param: initialValueArray: is the first guess for the root
        :param: accuracy: the highest tolerable value of error
        :returns:
            the closest (depending on the accuracy) values for theta1 and theta2 that minimize the residual error
            "i" The number of iterations until the final result

            "error" An error estimator in case the algorithm has gone wrong
        """

        # Simple empiric law
        if alpha>1.5 or alpha<.5:
            alpha=1

        fixedStepGradientMethodThetaOneValues = []
        fixedStepGradientMethodThetaTwoValues = []

        xCoordinates = self.TwoArmedRobotSystem.xCoordinates
        yCoordinates = self.TwoArmedRobotSystem.yCoordinates
        numberOfSamples = len(xCoordinates)

        for j in range(numberOfSamples):
            self.TwoArmedRobotSystem.p_x = xCoordinates[j]
            self.TwoArmedRobotSystem.p_y = yCoordinates[j]

            theNthValueOfThetaArray, i, error = self.getMinimaViaFixedStepGradForOneValue(initialValueArray, accuracy, maxIter, alpha)

            fixedStepGradientMethodThetaOneValues.append(theNthValueOfThetaArray[0])
            fixedStepGradientMethodThetaTwoValues.append(theNthValueOfThetaArray[1])

        return fixedStepGradientMethodThetaOneValues, fixedStepGradientMethodThetaTwoValues, i, error