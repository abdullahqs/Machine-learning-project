import numpy as np
import mlp
fid=open("lfe2.txt",'a')
a = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])

p = mlp.mlp(a[:,0:2],a[:,2:],6,beta=1,momentum=0.9,outtype='logistic')
fid.write(str("\n"))
fid.write(str("intial wiegths"))
fid.write(str("\n"))
fid.write('{}'.format(p.name()))
fid.write(str("\n"))
fid.write(str("before training"))
fid.write(str("\n"))


fid.write(str("\n"))
fid.write(str("matrix before training"))
fid.write(str("\n"))
fid.close()

p.confmat(a[:,0:2],a[:,2:])
fid=open("lfe2.txt",'a')
fid.write(str("\n"))
fid.write(str("after training"))
fid.write(str("\n"))
  
fid.close()


p.mlptrain(a[:,0:2],a[:,2:],0.03,4000)
fid.close()
fid=open("lfe2.txt",'a')
fid.write(str("wiegths after training"))
fid.write(str("\n"))
fid.write('{}'.format(p.name()))
fid.write(str("\n"))
fid.write(str("matrix after training"))
fid.write(str("\n"))
fid.close()
p.confmat(a[:,0:2],a[:,2:])
