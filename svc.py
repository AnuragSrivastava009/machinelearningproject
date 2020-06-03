import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use("ggplot")

d={-1:np.array([[1,9],[2,4],[3,6]]),   1:np.array([[5,7],[7,10],[6,8]])}


class SVM:
    def __init__(self,visualisation=True):#visualisation is for graphical output (parameter)
        self.vis=visualisation#storing the visualisation data in class var vis
        self.color={-1:"b",1:"g"}#assiging the colours for the groups
        if self.vis:
            fig=plt.figure()#creating the figure for plot
            self.ax=fig.add_subplot(1,1,1)#adding a subplot with 1,1,1 means 1 plot
    def fit(self,data):
        self.data=data       #data from the user is stored in class variable data
        #||w||:{w,b}
        self.opt_dict={}     # for storing the opted W and from our equation
        trans=[[1,1],[-1,-1],[1,-1],[-1,1]]     #to check variance of data of u and w
        all_data=[]     #to include every vector(point) data in 1-d frame
        for grp in self.data:     #to pick group from data
            for featset in self.data[grp]:    #to pick values from groups respectively
                all_data.extend(featset)  #to pick data from 2-d frame and extend will add one by one value to our list
        print(all_data)    #to see the complete data points in 1-d list
        self.maxfeat=max(all_data)   #max of the 1-d list of point
        self.minfeat=min(all_data)    #min of the 1-d list of points
        all_data=None    #dump the data as we have got the min and max b/w which the hyper plane might exists 
        step_sizes=[self.maxfeat*0.1,self.maxfeat*0.01,self.maxfeat*0.001]#step size for optimization of w higher ,lower,lowest
        print(step_sizes)
        b_rangem_start_stop=5   #start and stop feature set values(multiple) for range of b
        b_m_step=5   #stepsize multiple of b(try by removing this to view time complexity)
        latestopt=self.maxfeat*5   #max feature based latest optimum value for w
        for stp in step_sizes:    #step sizes of w
            w=np.array([latestopt,latestopt])#first assumed w base on latest optimum
            print(w)                   
            optimized=False
            while not optimized:
              for  b in np.arange(-1*(self.maxfeat*b_rangem_start_stop),(self.maxfeat*b_rangem_start_stop),(stp*b_m_step)):
                  for tr in trans:
                    w_t=w*tr # if the values are positive then the state of w is [+1,+1]
                    found_option=True
                    #yi(xi.w+b)
                    for yi in self.data:#yi is the class
                        for xi in self.data[yi]:#xi is the value of points in that group
                            if  yi*(np.dot(xi,w_t)+b)>=1:
                               pass
                            else:
                                found_option=False
                                break
                    if found_option:
                        self.opt_dict[np.linalg.norm(w_t)]=[w_t,b] #storing the w and b based on mangnitude of w in opt dict
              if w[0]<0:#if the values of w is lesser than 0 then need to change the step size(its the local minimum)
                    optimized=True
                    print("optimized")
              else:#decrement the w based on to the step size assumption
                    w=w-stp
            norms=min(self.opt_dict)#find the min magnitude of w
            opt_choice=self.opt_dict[norms]#fetch the value of lowest magnitude i.e w,b
            latestopt=opt_choice[0][0]+stp*5
            print("latest w and b",opt_choice[0],opt_choice[1],"terminated")
            for yi in self.data:
                for xi in self.data[yi]:
                    print(xi,"|",yi*(np.dot(opt_choice[1],xi))+opt_choice[1])
        self.w=opt_choice[0]#[w,b] hence the index of 0
        self.b=opt_choice[1]            
    def predict(self,feat):
        #sign(x.w+b)
        val=np.sign(np.dot(np.array(feat),self.w)+self.b)
        if val!=0 and self.vis:
                    self.ax.scatter(feat[0],feat[1],marker='*',s=400,c=self.color[val])
        print(val)
        return val
    def visual(self):
        for i in self.data:
            for x in self.data[i]:
                self.ax.scatter(x[0],x[1],s=100,color=self.color[i])
        
        def hyper(x,w,b,v):
            return (-w[0]*x-b+v)/w[1]
        hypx_min=self.minfeat
        hypx_max=self.maxfeat
        #positive plot
        psv1=hyper(hypx_min,self.w,self.b,1)
        psv2=hyper(hypx_max,self.w,self.b,1)
        self.ax.plot([hypx_min,hypx_max],[psv1,psv2],'k')
        #-ve plot
        nsv1=hyper(hypx_min,self.w,self.b,-1)
        nsv2=hyper(hypx_max,self.w,self.b,-1)
        self.ax.plot([hypx_min,hypx_max],[nsv1,nsv2],'k')
        #hyperplane
        sv1=hyper(hypx_min,self.w,self.b,0)
        sv2=hyper(hypx_max,self.w,self.b,0)
        self.ax.plot([hypx_min,hypx_max],[sv1,sv2],'k')
        plt.show()

d={-1:np.array([[1,9],[2,4],[3,6]]),
   1:np.array([[5,7],[7,10],[6,8]])}
s=SVM()
s.fit(data=d)
print(s.opt_dict)
pr=[[0,10],[1,3],[3,4],[3,5],[5,5],[6,-5]]
for p in pr:
    s.predict(p)
s.visual()
    
