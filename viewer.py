import re
import matplotlib.pyplot as plt

if __name__ == "__main__":


    robot_poses_x0 = []     
    robot_poses_y0 = []
    with open ('tutorial_before_rand.txt', 'rt') as myfile: 
        for myline in myfile:    
            word = 'VERTEX_SE3:EXPMAP'
            if re.search(word, myline):
                x = re.split("\s", myline)       
                robot_poses_x0.append(float(x[4]))   
                robot_poses_y0.append(float(x[2]))      
                print(x)   

    l_poses_x0 = []     
    l_poses_y0 = []
    with open ('tutorial_before_rand.txt', 'rt') as myfile: 
        for myline in myfile:    
            word = 'VERTEX_TRACKXYZ'
            if re.search(word, myline):
                x = re.split("\s", myline)       
                l_poses_x0.append(float(x[4]))   
                l_poses_y0.append(float(x[2]))      
                print(x)  

    robot_poses_x1 = []     
    robot_poses_y1 = []
    with open ('tutorial_after_rand.txt', 'rt') as myfile: 
        for myline in myfile:    
            word = 'VERTEX_SE3:EXPMAP'
            if re.search(word, myline):
                x = re.split("\s", myline)       
                robot_poses_x1.append(float(x[4]))   
                robot_poses_y1.append(float(x[2]))      
                print(x)   

    l_poses_x1 = []     
    l_poses_y1 = []
    with open ('tutorial_after_rand.txt', 'rt') as myfile: 
        for myline in myfile:    
            word = 'VERTEX_TRACKXYZ'
            if re.search(word, myline):
                x = re.split("\s", myline)       
                l_poses_x1.append(float(x[4]))   
                l_poses_y1.append(float(x[2]))      
                print(x)  

    '''robot_poses_x1 = []     
    robot_poses_y1 = []
    with open ('tutorial_after_rand2.txt', 'rt') as myfile: 
        for myline in myfile:    
            word = 'TUTORIAL_VERTEX_SE2'
            if re.search(word, myline):
                x = re.split("\s", myline)       
                robot_poses_x1.append(float(x[2]))   
                robot_poses_y1.append(float(x[3]))      
                print(x)    

    l_poses_x1 = []     
    l_poses_y1 = []
    with open ('tutorial_after_rand3.txt', 'rt') as myfile: 
        for myline in myfile:    
            word = 'TUTORIAL_VERTEX_SE2'
            if re.search(word, myline):
                x = re.split("\s", myline)       
                l_poses_x1.append(float(x[2]))   
                l_poses_y1.append(float(x[3]))      
                print(x)  
    '''

  

    '''
    plt.plot(robot_poses_x0, robot_poses_y0, 'r*', robot_poses_x1, robot_poses_y1 , 'b*', l_poses_x1, l_poses_y1, 'gx', l_poses_x2, l_poses_y2, 'rx')
    #plt.legend(["robot pose before", "robot pose after", "landmarks before"])
    plt.legend(["itr:0", "itr:7-1", "itr:7-2", "itr:14"])
    plt.title("600 node")
    plt.show()            
    '''


    plt.plot(robot_poses_x0, robot_poses_y0, 'ro', l_poses_x0, l_poses_y0, 'g*', robot_poses_x1, robot_poses_y1, 'b*', l_poses_x1, l_poses_y1, 'bo')
    #plt.legend(["robot pose before", "robot pose after", "landmarks before"])
    plt.legend(["robot0", "landmark0", "robot1", "landmark1"])
    #plt.title("600 node")
    plt.show()  