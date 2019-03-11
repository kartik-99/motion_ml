import numpy as np
import csv
x = []

f = open(r'''/home/daanvir/gg/project/SRM_project/motion_ml/ddd_data.txt''', 'r')

a = f.read().split('\n')
i = 0
y = []
print(len(a)/64)

while(i < len(a)-64*5):
    l = a[i].split(' ')
    if(i % 64 == 0):
        y.append(int(l[1]))
        a.pop(i)
    i += 1
q = []
u = 0
k = 0
y.pop(1024)
y = np.array(y)
print(y)
print(y.shape)
while(u < 1024):
    tan = []
    d=[]
    x = []
    for i in range(128):

        x.append(a[k].split(' '))

        l = []
        for j in range(9):
            x[i][j] = float(x[i][j])
            if(j % 3 == 0 and j != 0):
                ang = x[i][j-3]**2+x[i][j-2]**2+x[i][j-1]**2
                if(ang == 0):
                    l.append(np.pi/2)
                    continue
                l.append(np.arctan(x[i][j-2]/(ang)**(0.5)))
        j = j+1
        ang = x[i][j-3]**2+x[i][j-2]**2+x[i][j-1]**2
        if(ang == 0):
            l.append(np.pi/2)
        else:
            l.append(np.arctan(x[i][j-2]/(ang)**(0.5)))
        tan.append(l)

        x[i].pop(j)
        k += 1
    k = k-64

    o = np.average(x, axis=0)
    o = np.append(o, np.std(x, axis=0))
    o = np.append(o, np.amin(x, axis=0))
    o = np.append(o, np.amax(x, axis=0))
    o = np.append(o, np.average(tan, axis=0))
    o = np.append(o, np.std(tan, axis=0))
    o = np.append(o, np.amin(tan, axis=0))
    o = np.append(o, np.amax(tan, axis=0))

    if(u == 0):
        q = o
    else:
        q = np.vstack([q, o])
    
    u += 1
d=np.array(q)
print(d)


#with open("/home/daanvir/gg/project/SRM_project/x.csv","w+") as my_csv:
#    csvWriter = csv.writer(my_csv,delimiter=',')
#    csvWriter.writerows(d)
np.savetxt(r'/home/daanvir/gg/project/SRM_project/x.csv',
            d.flatten(), header=str(d.shape))
np.savetxt(r'/home/daanvir/gg/project/SRM_project/y.csv', y, delimiter=',')
print(d.shape)

