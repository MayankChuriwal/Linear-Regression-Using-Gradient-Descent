# Linear Regression using Gradient Descent

from numpy import * 

def compute_error_for_line_given_points(b, m, points):
    #Initialize it at 0
    totalError = 0
    #for every point
    for i in range(0, len(points)):
        # Get X Value
        x = points[i, 0]
        # Get Y Value
        y = points[i, 1]
        # Get Difference, square it & add it to the total
        totalError += (y - (m * x + b))**2
        
    # Get the average
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learningRate):
    # Starting points for Gradient
    b_gradient = 0
    m_gradient = 0
    
    N = float(len(points))
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # Direction with respect to b and m
        # Computing Partial Derivatives of our Error Function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
        
    # Update our b and m values using our partial derivatives
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return(new_b, new_m)


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    #Starting b & m
    b = starting_b
    m = starting_m
    
    # Gradient Descent
    for i in range(num_iterations):
        # Update b & m with new more accurate b and mby performing gradient descent steps
        b, m = step_gradient(b, m, array(points), learning_rate)
    return[b, m]
    

def run():
    # Step 1 collect our data
    points = genfromtxt("data.csv", delimiter=",")
    
    # Step 2 defining our hyperparameters
    # how fast should our model should converge?
    learning_rate = 0.0001
    
    # y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    
    # Step 3 Training our model
    print("Starting the Gradient Descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    
    print("Ending point of Gradient Descent at b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    
    
if __name__ == '__main__':
    run()
