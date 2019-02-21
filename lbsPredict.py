import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Screening by trained LBS (Local Beta Screening)')
    parser.add_argument('--data', type=str, metavar='D', help='data to be screened (required)')
    parser.add_argument('--model', type=str, default='lbsmodel', metavar='M', help='filename of the LBS model trained. the default is lbsmodel.npz (optional)')
    parser.add_argument('--result', type=str, default='ScreenResult', metavar='R', help='filename to keep the result of screening, index of screened samples are kept. the default is ScreenResult.txt (optional)')
    args = parser.parse_args()
    if args.result[-4:]!='.txt':
        args.result=args.result+'.txt'
    if args.model[-4:]!='.npz':
        args.model=args.model+'.npz'
    model=np.load(args.model)
    subset=model['subset']
    #print(subset)
    center=model['center']
    radius=model['radius']
    Xtest=np.loadtxt(args.test,delimiter=',',dtype=np.float32)
    Xtest=Xtest[:,subset]
    if Xtest.ndim>1:
        t=np.sum((Xtest-center)**2,axis=1)
    else:
        t=(Xtest-center)**2
    ind=np.where(t<=radius)[0]
    print(ind)
    with open(args.result,'w+') as f:
        for item in ind:
            f.write("%d, " % item)

if __name__=='__main__':
    main()
