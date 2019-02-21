import numpy as np
import argparse
from multiprocessing import Pool

def choosenk(n,k):
    kk=np.min([k,n-k])
    if kk<2:
        if kk==0:
            if k==n:
                x=np.arange(0,n)
            else:
                x=[]
        else:
            if k==1:
                x=np.arange(0,n).reshape((-1,1))
            else:
                x=np.arange(0,n)
                x=np.tile(np.arange(0,n),(n-1,1)).T.reshape((n-1,n)).T
    else:
        n1=n+1
        m=1
        for i in np.arange(1,kk+1):
            m=m*(n-kk+i)/i
        x=np.zeros((int(m),k),dtype=np.int)
        f=n1-k
        x[0:f,k-1]=np.arange(k-1,n).reshape((-1,))
        for a in np.arange(k-1,0,-1):
            d,h=f,f
            x[0:f,a-1]=a-1
            for b in np.arange(a+1,a+n-k+1):
                d=int(d*(n1+a-b-k)/(n1-b))
                e=f+1
                f=int(e+d-1)
                x[(e-1):f,a-1]=b-1
                x[(e-1):f,a:k]=x[(h-d):h,a:k]       
    return x

def bnnlearndisFaster(X,Y):
    Xc=np.unique(X[np.where(Y==1)[0],:],axis=0)
    fm=np.arange(1,X.shape[0]+1)+2
    smax0=0
    for ux in Xc:
        t=np.sum((X-ux)**2,axis=1)
        ti=t.argsort()
        s=(np.cumsum(Y[ti])+1)/fm
        t=t[ti]
        s[0:-1][t[1:]==t[0:-1]]=0
        smax=np.max(s)
        if smax>smax0:
            smax0=smax
    return smax0

def bnnlearndisFasterM(X,Y):
    Xc=np.unique(X[np.where(Y==1)[0],:],axis=0)
    fm=np.arange(1,X.shape[0]+1)+2
    smax0,dmax0,xmax0=0,0,0
    for ux in Xc:
        t=np.sum((X-ux)**2,axis=1)
        ti=t.argsort()
        s=(np.cumsum(Y[ti])+1)/fm
        t=t[ti]
        s[0:-1][t[1:]==t[0:-1]]=0
        smax=np.max(s)
        j=np.argmax(s)
        dmax=t[j]
        if smax>smax0:
            smax0=smax
            dmax0=dmax
            xmax0=ux
    return (smax0,dmax0,xmax0)

def bnnlearndisFasterM1d(X,Y):
    Xc=np.unique(X[np.where(Y==1)[0]])
    fm=np.arange(1,X.shape[0]+1)+2
    smax0,dmax0,xmax0=0,0,0
    for ux in Xc:
        t=(X-ux)**2
        ti=t.argsort()
        s=(np.cumsum(Y[ti])+1)/fm
        t=t[ti]
        s[0:-1][t[1:]==t[0:-1]]=0
        smax=np.max(s)
        j=np.argmax(s)
        dmax=t[j]
        if smax>smax0:
            smax0=smax
            dmax0=dmax
            xmax0=ux
    return (smax0,dmax0,xmax0)

def pmaxcal(rmax,nfeatures):
    # to fullfil: max return, s.t. C(return,nfeatures)<=rmax
    pm=rmax
    for i in np.arange(1,nfeatures+1):
        pm=pm*i
    p0=np.ceil(pm**(1/i))
    j0=p0
    jm=1
    for i in np.arange(1,nfeatures+1):
        jm=jm*j0
        j0=j0-1
    if jm>pm:
        return int(p0-1)
    for j in np.arange(1,nfeatures+1):
        jm=jm*(p0+j)/(p0+j-nfeatures)
        if jm>pm:
            return int(p0+j-1)
        
class Engine():
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def __call__(self,c):
        return bnnlearndisFaster(self.X[:,c],self.Y)
    
def lbsmodel(X,Y,rmax,outfile,ncpu):
    itermax=20
    idx=np.where(np.sum(X[np.where(Y==1)[0],:],axis=0)>0)[0]
    X=X[:,idx]
    p=X.shape[1]
    print('iteration: '+str(1))
    smax=np.zeros(p)
    for i in np.arange(0,p):
        (a,b,c)=bnnlearndisFasterM1d(X[:,i],Y) 
        smax[i]=a
    sfr=np.argsort(-smax,kind='stable')
    smaxk=smax[sfr[0]]
    print('loss = '+str(1-smaxk))
    loss=smaxk
    ind=np.where(smax==smaxk)[0][0]
    subset=idx[ind]
    (a,radius,center)=bnnlearndisFasterM1d(X[:,ind],Y)
 
    pmax=pmaxcal(rmax,2)
    if pmax>p:
        pmax=p
    
    us=sfr[0:pmax]
    for k in np.arange(2,itermax):
        print('iteration: '+str(k))
        comb=us[choosenk(pmax,k)]
        if ncpu>=1:
            pool=Pool(ncpu)
        else:
            pool=Pool()
        engine=Engine(X,Y)
        smax=np.asarray(pool.map(engine,comb),dtype=np.float32)
        sfr=np.argsort(-smax,kind='stable')
        smaxk=smax[sfr[0]]
        print('loss = '+str(1-smaxk))
        if smaxk>loss:
            loss=smaxk
            ind=np.where(smax==smaxk)[0][0]
            subset=idx[comb[ind]]
            (a,radius,center)=bnnlearndisFasterM(X[:,comb[ind]],Y)

            if k==itermax-1:
                print("maximum iter reached!")
                print('loss= '+str(1-loss)+', subset= '+str(subset)+\
                      ', center= '+str(center)+', radius= '+str(radius))
                np.savez('lbsmodel.npz',loss=loss,subset=subset,center=center,radius=radius)
                return loss
        else:
            print('Training is finished.\nResult:')
            print(' loss= '+str(1-loss)+',\n subset= '+str(subset)+\
                  ',\n center= '+str(center)+',\n radius= '+str(radius))
            np.savez(outfile,loss=1-loss,subset=subset,center=center,radius=radius)
            return loss
        
        us,ind=np.unique(comb[sfr],return_index=True)
        us=us[np.argsort(ind)]
        pmax=pmaxcal(rmax,k+1)
        if pmax>p:
            pmax=p
        us=us[:pmax].astype(int)



def main():
    parser = argparse.ArgumentParser(description='Training by LBS')
    parser.add_argument('--data', type=str, metavar='D', help='dataset to be trained, the last column is regarded as label (required)')
    parser.add_argument('--ne', type=int, default=2000000, metavar='N', help='maximum number of feature subsets to be evaluated in each iteration, default is 2000000 (optional)')
    parser.add_argument('--out', type=str, default='lbsmodel', metavar='O', help='filename to keep the output of training, default is lbsmodel.npz (optional)')
    parser.add_argument('--cpu', type=int, default=0, metavar='C', help='the number of CPUs to use, default is to use all of CPUs available (optional)') 
    args = parser.parse_args()
    print('Reading the data...')
    data=np.loadtxt(args.data,delimiter=',',dtype=np.float32)
    print('Start training:')
    if args.out[-4:]!='.npz':
        args.out=args.out+'.npz'
        print(args.out)
    lbsmodel(data[:,:-1],data[:,-1],args.cmax,args.out,args.cpu)


if __name__=='__main__':
    main()
